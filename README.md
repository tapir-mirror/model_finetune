# MiraStral: Fine-tuned Mistral for Therapeutic Conversations

This project fine-tunes the Mistral-7B-Instruct model to create MiraStral, a specialized model for therapeutic conversations and personal profiling. By incorporating tool use and maintaining empathetic responses, MiraStral aims to provide more effective and contextually aware therapeutic interactions.

## Project Overview

MiraStral enhances the base Mistral model by training it on high-quality therapeutic conversations that demonstrate:
- Tool-augmented responses
- Empathetic understanding
- Structured therapeutic techniques
- Safe and ethical interaction patterns

## Why Fine-tune Mistral for Therapy?

### Advantages of the Base Model
- Mistral-7B-Instruct provides an excellent foundation due to:
  - Strong reasoning capabilities
  - Good context understanding
  - Efficient performance for its size
  - High-quality instruction following

### Value of Therapeutic Fine-tuning
1. **Enhanced Empathy**: Training on therapeutic conversations helps the model better recognize and respond to emotional cues
2. **Tool Integration**: The model learns to effectively use tools while maintaining therapeutic rapport
3. **Safety First**: Fine-tuning reinforces ethical boundaries and safety protocols
4. **Structured Support**: Incorporates therapeutic frameworks while maintaining natural conversation flow

## Project Structure

```
fine_tune/
├── main.py              # Main fine-tuning script
├── fine_tune.py         # Model configuration
├── reformat_data.py     # Data preprocessing
├── run_finetune.sh      # HPC job script
└── requirements.txt     # Dependencies
```

## Pipeline Stages

### 1. Data Preparation
- Located in `reformat_data.py`
- Processes conversation data from worker directories
- Converts various formats to consistent JSONL structure
- Handles tool calls and response formatting

### 2. Model Configuration
- Located in `fine_tune.py`
- Sets up Mistral-7B-Instruct with 4-bit quantization
- Configures tokenizer with special tokens
- Optimizes for memory efficiency

### 3. LoRA Fine-tuning
- Implements Low-Rank Adaptation for efficient training
- Preserves base model capabilities while adding therapeutic competencies
- Focuses on key attention layers for optimal learning

### 4. Training Process
- Managed by `main.py`
- Implements:
  - Gradient accumulation for stability
  - Learning rate scheduling
  - Regular checkpointing
  - Progress monitoring

## Output Structure

The fine-tuned model and training artifacts are saved in:
```
/data/scratch/$USER/mira_finetune/
├── results/            # Training checkpoints and logs
└── final_model/        # Final fine-tuned model
```

## Therapeutic Applications

MiraStral is designed to enhance therapeutic interactions by:

1. **Contextual Understanding**
   - Recognizes therapeutic contexts
   - Maintains conversation history awareness
   - Adapts responses to client needs

2. **Tool-Enhanced Support**
   - Integrates external resources seamlessly
   - Maintains therapeutic flow during tool use
   - Provides structured interventions when appropriate

3. **Safety and Ethics**
   - Recognizes crisis situations
   - Maintains appropriate boundaries
   - Provides clear disclaimers about AI limitations

4. **Therapeutic Techniques**
   - Implements active listening
   - Uses reflection and validation
   - Maintains therapeutic alliance

## Important Notes

- This model is intended as a research tool and therapeutic aid, not a replacement for human therapists
- All interactions should be monitored and supervised by qualified professionals
- Regular evaluation of model outputs for safety and effectiveness is essential

## Future Directions

- Integration with more specialized therapeutic tools
- Expansion of training data to cover more therapeutic approaches
- Development of evaluation metrics for therapeutic effectiveness
- Enhanced safety monitoring and intervention protocols

## Running the Fine-tuning

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Submit the HPC job:
```bash
qsub run_finetune.sh
```

## Acknowledgments

This project builds on the excellent work of:
- Mistral AI team for the base model
- The therapeutic community for conversation data
- HuggingFace for training infrastructure 