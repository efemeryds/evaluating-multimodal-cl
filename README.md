# multi-modal-continual-learning

## Used repositories

- The core code with some modifications is taken from the following repositories:
    1) https://github.com/JiazuoYu/MoE-Adapters4CL
    2) https://github.com/Thunderbeee/ZSCL/tree/main

- Paths to data folder need to be adjusted in *src/datasets/* files and also in the *src/args.py* file.

## Evaluation of continual learning models:

1. ### **Retrieval evaluation**
   2. assesses how well the model can retrieve relevant items
   3. the purpose is to measure the model’s ability to understand and relate multimodal data (like image-to-text or
      text-to-image retrieval) and to evaluate how well the model has learned to map images and text into a shared
      embedding space
   3. process:
      4. **image-to-text retrieval**: given an image, retrieve the most relevant text descriptions
      5. **text-to-image retrieval**: given a text description, retrieve the most relevant images
      6. the performance is usually measured using metrics like Recall@K
   4. objective:
      5. focuses on the model’s ability to retrieve relevant data within the same modality
2. ### **Transfer evaluation**
    3. measures how well the model can adapt its learned representations to new tasks or datasets
    4. the purpose is to assess the generalization capability of the model’s representations to new and diverse tasks
    5. process:
        6. fine-tune the pre-trained model on a new dataset or task like classification
        7. evaluate the performance on the new task using task-specific metrics like accuracy
    8. objective:
        9. focuses on the model’s ability to transfer its learned representations to new, often task-specific contexts

Both methods play an important role in assessing the performance of CL models:

- retrieval evaluation helps in understanding how well the model retains and uses its multimodal embedding space over
  time
- transfer evaluation helps in assessing the adaptability and robustness of the model’s learned features when applied to
  new and varied tasks, providing insights into the model's ability to generalize and prevent catastrophic forgetting

## Evaluation done so far

- The authors evaluate MTIL by utilizing the following metrics *Transfer*, *Average*,
  and *Last*.
    - The *Transfer* metric assesses the model’s
      zero-shot transfer capability on unseen data.
    - *Last* evaluates the model’s memorization ability on historical knowledge.
    - *Average* is a composite metric measuring the mean performance across “Transfer” and “Last”.

## New components

- This repository extends existing code with the zero-shot retrieval capability.

- In paper *Learning Transferable Visual Models From Natural Language Supervision* the authors check the zero-shot
  transfer performance of CLIP for both
  text and image retrieval on the Flickr30k and MSCOCO datset.

![clip_retrieval_results.png](img%2Fclip_retrieval_results.png)

