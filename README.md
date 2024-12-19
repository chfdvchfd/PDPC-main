# PDPC: Plant disease phenotype captioning via Semantic correction and trait description dependency grammar based on LLM

## paper

## Abstract
Agriculture is the foundation of global food security and quality of life, with staple crops such as rice, wheat, and maize meeting the dietary needs of the majority of the world's population.  These crops are vulnerable to diseases that can cause severe yield losses;  for instance, wheat rust disease results in annual losses exceeding $2.9 billion.  Accurate description of the phenotypic characteristics of plant diseases plays a crucial supportive role in diagnosis, which is essential to ensure food security.  Recognizing the pivotal role of image caption in disease diagnosis, this study introduces the PDPC framework, designed to improve the precision and richness of plant disease image descriptions.  Using an extensive descriptive corpus, syntactic analysis, and optimization of semantic structures to significantly improve the quality and generalization of disease descriptions.  Additionally, we have constructed a dataset comprising 20,943 image descriptions of plant disease characteristics for over 60 plant species and 300 diseases.  Experimental results demonstrate that the PDPC framework outperforms existing models in accurately describing the characteristics of plant disease.  The introduction of this innovative tool not only improves the accuracy of disease descriptions but also provides robust support for the intelligent diagnosis and management of plant diseases, paving the way for improved plant health and increased agricultural yields.

## Environment

- **Set up our environment**

  ```bash
  conda create -n your_env_name python=x.x
  conda activate your_env_name
  pip install -r reqirements.txt
  pip install -e .
  ```

## Predict
- **Run the code of testing:**

  ```bash
  cd /main
  python PDPC.py
  ```


## Validation

- **Run the code of validation:**

  ```bash
  cd /main
  python val.py
  ```

