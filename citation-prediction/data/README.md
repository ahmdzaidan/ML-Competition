# Dataset Description - Gammafest IPB 2025

This dataset consists of pairs of academic papers, each accompanied by bibliographic metadata such as title, abstract, and publication year. Each record represents a potential citation link where the first document (`paper`) cites the second document (`referenced_paper`).

The objective is to build a machine learning model to predict the value of **`is_referenced`**, determining whether the citation link actually exists (`1`) or not (`0`), based on the available information.

---

### File Structure

Please place the dataset files in the `data/` directory following this layout:

```
data/
├── papers/                   # Paper Database: Folder containing full-text documents in .txt format (filename corresponds to paper_id)
│   ├── <paper_id_1>.txt
│   ├── <paper_id_2>.txt
│   └── ...
├── papers_metadata.csv       # Complete metadata for each document
├── train.csv                 # Labeled document pairs with the 'is_referenced' ground truth
├── test.csv                  # Unlabeled document pairs to be predicted by participants
└── sample_submission.csv     # Sample submission format for test data predictions
```

---

### Metadata Features (`papers_metadata.csv`)

| Column | Description |
| :--- | :--- |
| `paper_id` | Unique identifier for the paper |
| `doi` | Digital Object Identifier (if available) |
| `title` | Title of the academic paper |
| `publication_year` | Year of publication |
| `publication_date` | Date of publication |
| `cited_by_count` | Number of times this document has been cited by other papers |
| `type` | Document type (e.g., *article*, *conference*, etc.) |
| `authors` | List of authors |
| `concepts` | Topics or categorization of the document |
