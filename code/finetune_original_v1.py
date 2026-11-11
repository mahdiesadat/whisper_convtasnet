import argparse
import functools
import os
import json
import shutil
import torch

# FIX(resume/torch2.6): torch.load defaults to weights_only=True since 2.6 and
# rng_state.pth contains numpy objects, so resuming dies in _load_rng_state with
# _pickle.UnpicklingError.  Our checkpoints are local and trusted.
_torch_load_orig = torch.load
def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load_orig(*a, **kw)
torch.load = _torch_load_compat

from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor,
                          EarlyStoppingCallback)
from model.modeling_whisper_sidecar import WhisperSidecarForConditionalGeneration


# FIX(v3): `Seq2SeqTrainer.prediction_step` forwards every field of the collated
# batch straight into `model.generate(**inputs)`.  During eval the collator adds a
# `path` field (used by evaluation.py to name per-utterance outputs), and
# `generate()` rejects it in `_validate_model_kwargs`:
#
#   ValueError: The following model_kwargs are not used by the model: ['path']
#
# `remove_unused_columns=False` means the Trainer keeps that field, so the crash
# happens on the very first evaluation.  We strip the non-model metadata here,
# which keeps evaluation.py's standalone DataLoader (and its `batch["path"]`
# usage) fully intact.
class SidecarSeq2SeqTrainer(Seq2SeqTrainer):
    # Keys the collator/trainer add that are *not* accepted by generate().
    _NON_MODEL_KEYS = {
        "path", "speakers", "train_split",
        "is_eval_dataset", "for_target_asr_eval",
        "clean_audios",          # LOSS_V3
    }

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        inputs = {
            k: v for k, v in inputs.items()
            if k not in self._NON_MODEL_KEYS
        }
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )

from utils.callback import SaveCheckpointCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.dataset import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments, strtobool
from metrics.compute_metrics import WhisperOverlapWERCalculator
from peft import LoraConfig, get_peft_model, PeftConfig, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# paths
add_arg("train_data", type=str, default="dataset/libri2mix_train.jsonl", help="path for training data")
add_arg("dev_data", type=str, default="dataset/libri2mix_dev.jsonl", help="path for dev data")
add_arg("base_model", type=str, default="openai/whisper-small", help="Whisper model")
add_arg("model_name", type=str, default=None, help="model name")
add_arg("output_dir", type=str, default="output/", help="save path")
add_arg("logging_steps", type=int, default=200, help="log every x steps")
add_arg("resume_from_checkpoint", type=str, default=None, help="checkpoint path for recovery training")
add_arg("lr_anneal_steps", type=int, default=0, help="FIX(resume): absolute step at which the LR reaches 0 (0 = disabled)")
add_arg("lr_anneal_start", type=int, default=0, help="FIX(resume): step the anneal starts from (default: read from trainer_state.json)")

# training settings
add_arg("fp16",  type=bool, default=True, help="if use fp16")
add_arg("max_steps", type=int, default=400000, help="max training steps")
add_arg("warmup_steps", type=int, default=500, help="warming up steps for lr")
add_arg("eval_steps", type=int, default=2000, help="eval every this many steps")
add_arg("save_steps", type=int, default=2000, help="save checkpoint every x steps")
add_arg("learning_rate", type=float, default=2e-4, help="learning rate")
add_arg("lr_scheduler_type", type=str, default="linear", help="learning rate scheduler")
add_arg("eval_delay", type=int, default=15, help="eval after x epoches")
add_arg("gradient_accumulation_steps", type=int, default=1, help="gradient accumulation")
add_arg("per_device_train_batch_size", type=int, default=16, help="training batch size")
add_arg("per_device_eval_batch_size", type=int, default=4, help="eval batch size")
add_arg("num_workers", type=int, default=8, help="Number of threads for reading data")
add_arg("language", type=str, default="en", help="en or zh. if it is None, it is trained to be multilingual")
add_arg("timestamps", type=strtobool, default='false', help="whether to use timestamp data during training")
add_arg("min_audio_len", type=float, default=0.5,  help="min audio length, in seconds")
add_arg("max_audio_len", type=float, default=30, help="max audio length, in seconds")
add_arg("local_files_only", type=bool, default=False, help="whether to only load the model locally")
add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'], help="whisper task")
add_arg("augment_config_path", type=str, default=None, help="data augmentation configuration file path")

# model custom config
add_arg("use_lora", type=strtobool, default='false', help="whether to train lora")
add_arg("use_adalora", type=strtobool,  default='false', help="whether use adalora instead of lora")
add_arg("num_spks", type=int, default=2, help="max number of speakers in the training set")
add_arg("sidecar_loc", type=int, default=1, help="location of sidecar")
add_arg("soft_prompt_len", type=int, default=4, help="soft prompt in decoder input")
add_arg("target_asr", type=strtobool, default='false', help="whether to train the target asr task")
# FIX(v2): sidecar architecture / wiring options exposed to the CLI.
# The old code hardcoded the separator to *replace* the encoder hidden states and
# to use a (hardcoded) PReLU mask activation, which is what produced the
# input-independent repetition loops.
add_arg("sidecar_residual", type=strtobool, default='true',
        help="FIX(v2): add a gated residual around the separator instead of replacing the encoder states")
add_arg("sidecar_activate", type=str, default="sigmoid", choices=['sigmoid', 'relu', 'gelu', 'prelu', 'softmax'],
        help="FIX(v2): mask activation inside the separator (sigmoid = Conv-TasNet standard, bounded masks)")
add_arg("mask_norm", type=str, default="none", choices=['none', 'softmax'],
        help="FIX(v2): optional cross-speaker mask normalisation inside the separator")
add_arg("sidecar_gate_init", type=float, default=0.1,
        help="FIX(v2): initial value of the learnable residual gate (0 would kill the separator gradient)")
# FIX(v2): inference-time anti-repetition controls (set on generation_config).
add_arg("repetition_penalty", type=float, default=1.15,
        help="FIX(v2): generation repetition penalty (1.0 disables it)")
add_arg("no_repeat_ngram_size", type=int, default=4,
        help="FIX(v2): generation no-repeat n-gram size (0 disables it)")
# FIX(v2): step-based evaluation/saving (the original run silently used epoch-based
# strategies so --eval_steps / --save_steps were ignored).
add_arg("eval_strategy", type=str, default="steps", choices=['steps', 'epoch'],
        help="FIX(v2): evaluation strategy")
add_arg("save_strategy", type=str, default="steps", choices=['steps', 'epoch'],
        help="FIX(v2): checkpoint save strategy")
# FIX(v4): disk / early-stopping controls (were hardcoded).
add_arg("save_total_limit", type=int, default=6,
        help="FIX(v4): max checkpoints kept on disk (~1.1 GiB each); 6 is safe")
add_arg("early_stopping_patience", type=int, default=0,
        help="FIX(v4): stop when dev WER has not improved for N evals (0 = disabled)")

args = parser.parse_args()
world_size = int(os.environ.get("WORLD_SIZE", 1))

# if 'whisper-' in args.base_model: 
#     model_name = f"{os.path.basename(args.base_model).split('whisper-')[1]}_{args.model_name}"
# else:
#     model_name = f"{args.model_name}_{args.base_model.split('/')[-2]}"

model_name = args.model_name
output_dir = os.path.join(args.output_dir, model_name)

try:
    peft_config = PeftConfig.from_pretrained(args.resume_from_checkpoint if args.resume_from_checkpoint is not None else args.base_model)
    args.use_adalora = True if peft_config.peft_type == "ADALORA" else args.use_adalora
    base_base_model = peft_config.base_model_name_or_path
    print("The base model is with LoRA.")
except:
    base_base_model = args.base_model
    peft_config = None


# Get WhisperProcessor, which includes cnn feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained(base_base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=True,
                                             local_files_only=args.local_files_only)
processor.save_pretrained(os.path.join(output_dir, "tokenizer"))

with open(os.path.join(output_dir, "finetune_config.json"), 'w') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path,
                              num_spks=args.num_spks,
                              )
dev_dataset = CustomDataset(data_list_path=args.dev_data,
                             processor=processor,
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len,
                             num_spks=args.num_spks,
                             is_eval_dataset=True
                             )

device_map = "auto"
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# load model
model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
model = WhisperSidecarForConditionalGeneration.from_pretrained(model_path,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only,
                                                        sidecar_loc=args.sidecar_loc,
                                                        num_spks=args.num_spks,
                                                        soft_prompt_len=args.soft_prompt_len,
                                                        target_asr=args.target_asr,
                                                        # FIX(v2): new separator wiring
                                                        sidecar_residual=(args.sidecar_residual),
                                                        sidecar_activate=args.sidecar_activate,
                                                        mask_norm=args.mask_norm,
                                                        sidecar_gate_init=args.sidecar_gate_init,
                                                        attn_implementation="sdpa",)

modules_to_save = ['sidecar', 'sep_enc', 'sep_dec', 'soft_prompt_embeds', 'proj_target']
print(f"Modules to train and save: {modules_to_save}")

if peft_config is not None:
    print(f'loading LoRA modules... if_train_lora = {args.use_lora}')
    peft_config.modules_to_save = modules_to_save
    model = PeftModel.from_pretrained(model, model_path, is_trainable=True if args.use_lora else False, config=peft_config)
elif args.use_lora:
    print(f'adding LoRA modules...')
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    print(target_modules)
    if args.use_adalora:
        peft_config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                    lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules, modules_to_save=modules_to_save)
    else:
        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules,
                                    lora_dropout=0.05, bias="none", modules_to_save=modules_to_save)
else:
    # init model parameters
    for name, params in model.named_parameters():
        params.requires_grad = False
        if any(e in name for e in modules_to_save) or ("lora" in name and args.use_lora):
            params.requires_grad = True

if not (args.resume_from_checkpoint or "sidecar" in args.base_model):
    # FIX(v2): the old call `model.param_init(modules_to_save)` re-initialised *every*
    # parameter whose name contained one of the module substrings (normal_(0, 0.02)),
    # which clobbered the LayerNorm / GlobalLayerNorm scale (must be 1.0) and
    # the PReLU slope (must be 0.25) inside the separator.  A ~0.02 scale compounded
    # through 24 conv blocks pushes the separator output to 0 (exactly 0.0 in fp16),
    # so the frozen decoder cross-attends to all-zero keys and free-runs on the LM
    # prior -> word-repetition loops, no EOS, WER ~400%.
    # `init_sidecar_weights` does a type-aware re-init of only the *new* modules and
    # then restores the residual gate.
    model.init_sidecar_weights(gate_init=args.sidecar_gate_init)
    print(f"FIX(v2): sidecar re-initialised (activate={args.sidecar_activate}, "
          f"residual={args.sidecar_residual}, mask_norm={args.mask_norm}, "
          f"gate_init={args.sidecar_gate_init})")


# setting for decoder soft_prompt
if args.soft_prompt_len > 0:
    if not hasattr(model.generation_config, "prev_sot_token_id"):
        model.generation_config.prev_sot_token_id = 50361
    model.generation_config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    model.config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    force_list = list(range(100, 100+args.soft_prompt_len )) + [startoftranscript_id] + [i[1] for i in processor.get_decoder_prompt_ids()]
    model.config.forced_decoder_ids = [(i+1, x) for i, x in enumerate(force_list)]
else:
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    model.generation_config.decoder_start_token_id = startoftranscript_id
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()
print(f"Forced decoding tokens: {model.config.forced_decoder_ids}")

model.config.suppress_tokens = []
model.generation_config.max_new_tokens = 200

# FIX(v2): anti-repetition controls.  The original config had NO repetition_penalty and
# NO no_repeat_ngram_size, so an unconstrained decoder loops forever.  These are set on
# the generation_config object that `generate()` actually reads.
model.generation_config.repetition_penalty = args.repetition_penalty
if args.no_repeat_ngram_size > 0:
    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
model.generation_config.early_stopping = False
model.generation_config.length_penalty = 1.0
print(f"FIX(v2): generation_config repetition_penalty={model.generation_config.repetition_penalty}, "
      f"no_repeat_ngram_size={getattr(model.generation_config, 'no_repeat_ngram_size', None)}")

# FIX(v2): fail fast if the separator is dead (all-zero masks / zero hidden-state variance).
# This is exactly the failure mode that produced the repetition loops.
if world_size == 1 or int(os.environ.get("LOCAL_RANK", 0)) == 0:
    try:
        from tools.sanity_check import check_sidecar_is_alive
        check_sidecar_is_alive(model, num_spks=args.num_spks, verbose=True)
    except Exception as e:
        print(f"FIX(v2): sanity check skipped/failed: {e}")


# Register forward, otherwise the multi-GPU training will fail.
model.get_encoder().conv1.register_forward_hook(make_inputs_require_grad)


# training args
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,
                             per_device_train_batch_size=args.per_device_train_batch_size,
                             per_device_eval_batch_size=args.per_device_eval_batch_size,
                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                             learning_rate=args.learning_rate,
                             lr_scheduler_type=args.lr_scheduler_type,
                             warmup_steps=args.warmup_steps,
                             max_steps=args.max_steps,
                             fp16=args.fp16,
                             save_strategy=args.save_strategy,
                             save_steps=args.save_steps,
                             save_total_limit=args.save_total_limit,
                             evaluation_strategy=args.eval_strategy,
                             eval_steps=args.eval_steps,
                             eval_delay=args.eval_delay,
                             load_best_model_at_end=False,
                             report_to=["tensorboard"],
                             torch_compile=False,
                             optim='adamw_torch',
                             ddp_find_unused_parameters=False if ddp else None,
                             dataloader_num_workers=args.num_workers,
                             logging_steps=args.logging_steps,
                             remove_unused_columns=False,
                             label_names=["labels"],
                             predict_with_generate=True,
                             metric_for_best_model="wer",
                             greater_is_better=False,
                            )

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print(model)
    print_arguments(args)
    print(f"world_size: {world_size}")
    print(f"Model Name: {model_name}")
    print(f"Training data: {len(train_dataset)}, Eval data: {len(dev_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ", 
          f"All parameters: {sum(p.numel() for p in model.parameters())}")
    print("------------------------------------------------")


wer_cal = WhisperOverlapWERCalculator(processor=processor, num_spks=args.num_spks, soft_prompt_len=args.soft_prompt_len)


# LOSS_V3: hand Whisper's exact mel filterbank to the model (GPU mel for the clean path)
if float(os.environ.get("SIDECAR_REC_LAM", "0") or 0) > 0:
    _shape = model.set_mel_filters(processor.feature_extractor.mel_filters)
    print("LOSS_V3: mel filters set", _shape,
          "| sep=", os.environ.get("SIDECAR_SEP_LAM"), os.environ.get("SIDECAR_SEP_MODE", "stream_cos"),
          "| rec=", os.environ.get("SIDECAR_REC_LAM"), os.environ.get("SIDECAR_REC_MODE", "dist"),
          "| pit=", os.environ.get("SIDECAR_REC_PIT", "tied"))

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, num_spks=args.num_spks, soft_prompt_len=args.soft_prompt_len, target_asr=args.target_asr)
# P3 gen2 bootstrap: bind the FP32 centre once at startup (never from a resumed ckpt)
if float(os.environ.get("SIDECAR_P3_BETA", "0") or 0) > 0:
    if not getattr(model.config, "p3_center_sha256", None):
        _cp = os.environ.get("SIDECAR_P3_CENTER")
        if not _cp or not os.path.isfile(_cp):
            raise RuntimeError("P3_BETA>0 but no SIDECAR_P3_CENTER to bootstrap from")
        model.bind_p3_center(torch.load(_cp, map_location="cpu").float(),
                             float(os.environ["SIDECAR_P3_MARGIN"]))
        print("P3 bootstrap: centre bound from", _cp)
    model.validate_p3_center()

trainer = SidecarSeq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=dev_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SaveCheckpointCallback] + (
                             [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
                             if args.early_stopping_patience > 0 else []
                         ),
                        #  preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         compute_metrics=wer_cal
                        )

# FIX(resume): the stock resume path either restores the OLD scheduler (base_lrs
# 2e-4, last_epoch from the linear run) or -- if scheduler.pt is missing --
# builds a cosine whose horizon is max_steps counted from zero, so the LR jumps
# up and never anneals.  We keep both checkpoint files so optimizer.pt (Adam
# moments) is restored, then swap in a LambdaLR anchored at the checkpoint's
# real global step (create_scheduler runs before trainer_state.json is read, so
# self.state.global_step is still 0 there -- hence the explicit anchor).
if getattr(args, "lr_anneal_steps", 0) and args.lr_anneal_steps > 0:
    import math, types
    from torch.optim.lr_scheduler import LambdaLR

    _HORIZON = int(args.lr_anneal_steps)
    _ANCHOR = int(getattr(args, "lr_anneal_start", 0) or 0)

    def _anneal_start(self, checkpoint=None):
        s = 0
        if checkpoint:
            ts = os.path.join(checkpoint, "trainer_state.json")
            try:
                if os.path.isfile(ts):
                    with open(ts, encoding="utf8") as _f:
                        s = int(json.load(_f).get("global_step", 0) or 0)
            except Exception as _e:
                print(f"FIX(resume): could not read trainer_state.json: {_e}")
        if s <= 0:
            s = _ANCHOR
        if s <= 0:
            s = int(getattr(self.state, "global_step", 0) or 0)
        return s

    def _install_anneal(self, checkpoint=None):
        start = _anneal_start(self, checkpoint)
        if start >= _HORIZON:
            print(f"FIX(resume): global_step={start} >= horizon={_HORIZON}, no anneal")
            return
        # optimizer.pt carries the OLD param_groups (lr and initial_lr = 2e-4),
        # so both must be overwritten or the 1.0 factor would yield 2e-4.
        for g in self.optimizer.param_groups:
            g["lr"] = self.args.learning_rate
            g["initial_lr"] = self.args.learning_rate

        def _factor(step):
            if step <= start:
                return 1.0
            if step >= _HORIZON:
                return 0.0
            p = (step - start) / float(_HORIZON - start)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        self.lr_scheduler = LambdaLR(self.optimizer, _factor, last_epoch=max(start, 0))
        print(f"FIX(resume): cosine {start} -> {_HORIZON} "
              f"(lr now {self.lr_scheduler.get_last_lr()[0]:.3e} -> ~0)")

    _orig_create = trainer.create_scheduler
    _orig_load = trainer._load_optimizer_and_scheduler

    def _create_scheduler(self, num_training_steps, optimizer=None):
        _orig_create(num_training_steps, optimizer)
        _install_anneal(self, None)

    def _load_and_anneal(self, checkpoint):
        _orig_load(checkpoint)
        _install_anneal(self, checkpoint)

    trainer.create_scheduler = types.MethodType(_create_scheduler, trainer)
    trainer._load_optimizer_and_scheduler = types.MethodType(_load_and_anneal, trainer)

model.config.use_cache = True
model.generation_config.use_cache = True
trainer._load_from_checkpoint = load_from_checkpoint

# start training
from p3_production_runtime import install as install_p3_production
install_p3_production(trainer)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# save the last checkpoint
trainer.save_state()


# FIX(v4): HF checkpoint dirs do NOT contain the tokenizer / feature-extractor files,
# so WhisperProcessor.from_pretrained(checkpoint_dir) fails.  Copy them in.
def _embed_tokenizer(ckpt_dir):
    src_dir = os.path.join(output_dir, "tokenizer")
    if not (os.path.isdir(ckpt_dir) and os.path.isdir(src_dir)):
        return
    for fn in os.listdir(src_dir):
        s = os.path.join(src_dir, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(ckpt_dir, fn))
    print(f"FIX(v4): tokenizer copied into {ckpt_dir}")


if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    _embed_tokenizer(os.path.join(output_dir, "checkpoint-final"))
    _embed_tokenizer(os.path.join(output_dir, "checkpoint-best"))
