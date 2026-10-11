- AdamW is a deep learning optimization algorithm that fixes standard Adam by **decoupling weight decay** from the gradient-based update step.
Key diff from Adam:
  - Weight Decay Handling: In standard Adam, L2 regularization is added directly to the loss function, meaning the penalty gets scaled by Adam’s adaptive per-parameter learning rates. **AdamW computes the standard Adam update from the loss gradient first, and then shrinks the weights separately as a distinct step**
  - Better Generalization: Because weight decay is decoupled from the historical gradient moments (\(m_t, v_t\)), the optimal choice of weight decay and learning rate do not interfere with each other.
  - Transformer Standard: AdamW has become the default optimizer for training large language models (such as GPT) and modern transformer architectures.
  
- warmup usually refers to learning rate warmup, where you start training with a very small learning rate and gradually increase it to your target learning rate over an initial number of steps or epochs. 

- A cosine learning rate (lr) scheduler **smoothly decreases the learning rate** following the shape of a cosine half-wave from a maximum value down to a minimum value.

- Pydantic – Request/response validation

- Automatic Mixed Precision (AMP) is a technique used in deep learning to speed up model training and reduce memory usage by automatically switching between 16-bit and 32-bit floating-point numbers.

- an initial value given to a pseudo-random number generator (RNG) to ensure that random processes **produce the exact same sequence** of numbers across multiple runs.