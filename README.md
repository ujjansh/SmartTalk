# SmartTalk: Toward Practical LLM-Driven Control for Smart-Home IoT Systems

## Paper

SmartTalk explores how large language models can interpret everyday household
instructions and translate them into structured actions for smart-home IoT
devices. Instead of producing an open-ended text response, the model generates
JSON identifying the target rooms, devices, and requested states.

The paper brings together three complementary approaches:

- **QLoRA fine-tuning:** adapts compact language models to smart-home commands,
  room and device vocabulary, and the required JSON output format.
- **Structured pruning and quantization:** compresses larger models by removing
  selected attention and MLP structures and using low-bit representations for
  edge deployment.
- **Retrieval-augmented personalization:** supplies household-specific shortcuts,
  aliases, preferences, and routines through context retrieved from a local
  knowledge base.

The proposed pipeline combines the user's command with relevant household
context, generates a structured response, and validates the proposed actions
before device control. SmartTalkBench provides command-action examples for
studying this task, including direct instructions and broader household routines.

## Code

The [code folder](code/) contains the released scripts for model
adaptation, compression, and inference on the smart-home command-generation task.

| File                                                                             | Description                                                               |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [smarttalk_qlora_finetune.py](code/smarttalk_qlora_finetune.py)                   | QLoRA fine-tuning and evaluation for command-to-JSON generation           |
| [smarttalk_structured_prune.py](code/smarttalk_structured_prune.py)               | Structured pruning of attention and MLP components, with model evaluation |
| [smarttalk_jetson_prune_runner.py](code/smarttalk_jetson_prune_runner.py)         | Wrapper for running the structured-pruning workflow on Jetson             |
| [smarttalk_gguf_eval_100.py](code/smarttalk_gguf_eval_100.py)                     | Runs GGUF inference and compares generated JSON with reference outputs    |

The scripts expose command-line options for selecting models, supplying the
dataset, and choosing output locations. The training and evaluation scripts
consume the CSV dataset. GGUF inference uses a separately supplied model and
llama.cpp runtime.

## Dataset

The [dataset folder](dataset/) pairs natural-language smart-home commands with
reference JSON actions. The task covers explicit device instructions, implicit
intent, multi-device routines, and household-specific commands described in the
paper.

The available files are:

- [smart_home_100k_clean.csv](dataset/smart_home_100k_clean.csv): command-action
  pairs for the supplied training and evaluation scripts.
- [smart_home_100k_clean.json](dataset/smart_home_100k_clean.json): an equivalent
  JSON representation of the CSV records.
- [examples.json](dataset/examples.json): illustrative input/output pairs for
  inspecting the expected structure.

Each record has an `input` containing the user's command and an `output` mapping
room names to devices and their requested states. CSV stores the output as a
serialized JSON string; the JSON files store it as a nested object.

Room names include `balcony`, `bathroom`, `bedroom`, `dining_room`, `hall`,
`kitchen`, `living_room`, and `study_room`. Device categories include `lights`,
`fan`, `tv`, `exhaust`, `ac`, `blinds`, `music_system`, `computer`, and `geyser`.

For example: 

```json
{
  "input": "Could you open the blinds in the bedroom?",
  "output": {
    "bedroom": {
      "blinds": "open"
    }
  }
}
```

**Authors:** Ujjansh Sundram, Arushika, Pankaj Choudhury, and Alakesh Kalita.
