# PyUMLS-Similarity

                            ╔════════════════════════════════════════════════════════╗
                            ║                    PYUMLS SIMILARITY                   ║
                            ║            Unified Medical Language System Tools       ║
                            ╚════════════════════════════════════════════════════════╝
**PyUMLS-Similarity** is a semantic similarity engine for biomedical concept analysis, built on the UMLS. It supports transformer-based embeddings, various semantic similarity metrics (e.g. path, LCH, Resnik, Lin), and interactive exploration of concept relationships. It aims to facilitate the integration of these methods into existing NLP workflows.

---

## Features

### UMLS Integration
- Connects to local UMLS MySQL databases (2024AB and earlier).
- Parses MRCONSO, MRREL, MRSTY, MRDEF, MRHIER; validates schema and vocabularies.

### Graph Construction
- Builds directed/undirected graphs from MeSH or SNOMED using `PAR`, `RB`, `RN`, `CHD`, and `is_a`.
- Supports hierarchical trees via `MRSAT.MN`.
- Visualizes paths with PyVis; optional Neo4j export for interactive exploration.

### Semantic Similarity
- Implements path-based metrics: Path Length, Wu-Palmer, LCH.
- Supports IC-based metrics: Resnik, Lin, Jiang-Conrath.
- Adds embedding-based methods: Word2Vec, GloVe, Transformer (e.g., PubMedBERT).
- Outputs are normalized and compatible with heatmap comparison.

### Concept Linking & LCS
- Resolves CUIs from terms or accepts direct input.
- Computes shortest paths and lowest common subsumers.
- Extracts STRs, definitions, synonyms, semantic types/groups.

### NER and Free Text Input
- Runs NER with Hugging Face models (e.g., `dslim/bert-base-NER`).
- Links extracted spans to UMLS CUIs and computes similarity between all pairs.

### Output & Visualization
- Saves results to CSV, Excel, or Parquet.
- Plots similarity heatmaps across selected metrics.
- Renders interactive graphs with hoverable concept details.

---

## Requirements and Installation

### Installation

You can install `PyUMLS-Similarity` via pip:

```bash
pip install PyUMLS-Similarity
```

You must also have access to a local UMLS installation and corresponding MySQL database. Additional setup steps are outlined below.


## UMLS Setup Requirements

To use this package, you must have a MySQL instance of the UMLS configured in your environment.  You can get a UMLS License at: https://www.nlm.nih.gov/research/umls/index.html

You can install the UMLS following the instructions here: https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html

It is recommended that you also follow modify your my.ini file for MySQL using the parameters here for optimal performance (https://www.nlm.nih.gov/research/umls/implementation_resources/scripts/README_RRF_MySQL_Output_Stream.html)

## Optional: Neo4j Integration

To explore UMLS concept graphs visually and semantically:

Install Neo4j and run a graph database locally or in the cloud. The engine supports exporting UMLS graphs to Neo4j (AuraDB) with rich concept metadata and relationship types.

<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/neo4j%20rn%205k.PNG" style="max-width:100%; height:auto;"/>
<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/neo4j%20rb%205k.PNG" style="max-width:100%; height:auto;"/>
<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/neo4j%20par%205k%202.PNG" style="max-width:100%; height:auto;"/>

## Required Files

  ### Semantic Group Mapping
  
  You will need the SemGroups.txt file, available at:
  
  https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/SemGroups.txt

  ### Information Content File
  
  To use IC-based similarity measures (Resnik, Lin, Jiang-Conrath), you must provide an information content (IC) file mapping CUIs to IC values.

  A precomputed IC file (cui_ic_pubmed.parquet) is included with the package based on ~72,000 CUIs using term frequency estimates from a large PubMed abstracts corpus:
  yanjx21/PubMed on Hugging Face. 
  
  ***Note: Information Content-based measures are dependent on the corpus used to determine term frequencies. As such, you may want to explore different corpora when exploring these metrics. The file provided here is for prototyping/experimentation purposes.***

  <img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/cuiic.png" style="max-width:100%; height:auto;"/>

## Basic Usage

Start by importing the module like so:

```python
from PyUMLS-Similarity import UMLSSemanticEngine
```

Then, you can use the routines below to get you started. Start by initiating the connection to the MySQL UMLS database. Load the SemGroups.txt file.  

You can also optionally load the transformer model you want to use for your embeddings and the cui_ic.parquet file if you are interested in information content-based metrics

```python
# Step 1: Set up MySQL connection info
mysql_info = {
    "host": "localhost",
    "user": "root",
    "password": "your_password",
    "database": "umls2024"
}

# Step 2: Instantiate the engine and load resources
umls_utils = UMLSSemanticEngine(mysql_info)
umls_utils.load_transformer_model(model_name="NeuML/pubmedbert-base-embeddings")
tui_to_group = umls_utils.load_semantic_group_mapping("SemGroups.txt")
umls_utils.load_cui_ic_from_parquet("ic/cui_ic_pubmed.parquet")
```

Next, you'll want to construct the networks for whatever ontology you are interested in. In the example below, I'm creating the graphs for the MeSH ontology using PAR relationships (in G_MSH) and including RB and RN relationships (in G_MSH_expanded). 

```python
# Step 3: Build graphs from UMLS MRREL
G_MSH = umls_utils.build_mesh_graph_from_mrrel()
UG_MSH = G_MSH.to_undirected()

G_MSH_expanded = umls_utils.build_mesh_graph_expanded()
UG_MSH_expanded = G_MSH_expanded.to_undirected()
```

Then, there are different inputs that can be used for the semantic calculations. You can either provide a list of tuples containing the concept pairs that you want to compare or you can provide a piece of text. Below is an example of a workflow using the tuple list.

```python
# Step 4: Define concept pairs
concept_pairs = [
    ('Heart', 'Lung'),
    ('Kidney', 'Liver'),
    ('Asthma', 'Influenza'),
    ('Diabetes Mellitus', 'Hypertension'),
    ('Stroke', 'Alzheimer Disease'),
    ('Breast', 'Prostate'),
    ('Blood', 'Platelets'),
    ('Vaccination', 'Antibody Formation'),
    ('MRI', 'Computed Tomography'),
    ('Anxiety', 'Depression'),
    ('Amoxicillin', 'Vancomycin'),
    ('Surgery', 'Wound Healing'),
    ('Fluoxetine', 'PTSD'),
    ('Breast Cancer', 'Prostate Cancer'),
]

# Step 5: Run semantic similarity analysis
df_results = umls_utils.generate_concept_path_analysis(
    concept_pairs=concept_pairs,
    UG=UG_MSH,
    UG_expanded=UG_MSH_expanded,
    tui_to_group=tui_to_group,
    sab="MSH",
    max_depth=18,
    similarity_metrics=["path", "wup", "lch", "transformer", "resnik", "lin"],
    verbose=False
)
```

To process a piece of text you can do so similarly as shown below:

```python
# Step 4: Load your NER model from HuggingFace
umls_utils.load_ner_model(model_name="HUMADEX/english_medical_ner")

# Step 5: Provide your piece of text
sample_text = """
The patient presented with symptoms of both asthma and influenza, experiencing shortness of breath and fever.
Prior history includes diabetes mellitus and hypertension, managed through insulin therapy and dietary changes.
Imaging techniques such as MRI and computed tomography were utilized to assess internal damage. Elevated blood pressure was noted alongside low platelet count.
A family history of breast cancer and prostate cancer was also reported. The patient received a vaccination and demonstrated antibody formation within two weeks.
Mood assessments revealed moderate anxiety and depression. A surgical plan was developed to accelerate wound healing post-intervention."
"""

# Step 6: Run semantic similarity analysis
df_results_ner = umls_utils.run_pipeline_on_text(
      sample_text,
      UG_MSH_expanded,
      tui_to_group,
      sab="MSH",
      verbose=True,
      similarity_metrics=["path", "wup", "lch"]
)
```

Regardless of the method, you'll end up with a dataframe that provides detailed information about each concept pair like their CUI, their STR value in the ontology, their DEF in the ontology, the exact path between them (if it exists), their lowest common subsumer, 
semantic groups, and several others as shown below.

<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/df.PNG" style="max-width:100%; height:auto;"/>

You can also visualize the similarity metrics via a heatmap using:

```python
umls_utils.plot_all_similarity_metrics_heatmap(df_results, max_depth=18, normalize=True)
```

<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/heatmap1.png" style="max-width:100%; height:auto;"/>

Or you can also visualize radar plots for the concept pairs as shown here:

```python
umls_utils.plot_all_similarity_metrics_radar(df_results, max_depth=18, normalize=True, plots_per_row=5)
```

<img src="https://raw.githubusercontent.com/victormurcia/pyramedic-umls/main/images/radar.png" style="max-width:100%; height:auto;"/>
