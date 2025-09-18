import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers

# --------------------------
# Gene Encoder
# --------------------------
class GeneEncoder:
    def __init__(self):
        # 20 standard amino acids
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i for i, aa in enumerate(self.amino_acids)}

    def protein_one_hot(self, protein: str) -> np.ndarray:
        """Convert protein sequence into one-hot encoding (len, 20)."""
        one_hot = np.zeros((len(protein), 20), dtype=np.float32)
        for i, aa in enumerate(protein):
            if aa in self.aa_to_int:
                one_hot[i, self.aa_to_int[aa]] = 1.0
        return one_hot

    def pad_one_hot(self, sequences, max_len=None):
        """Pad a list of 2D one-hot arrays to the same length (truncate if needed)."""
        feature_dim = sequences[0].shape[1]
        lengths = np.array([seq.shape[0] for seq in sequences])
        if max_len is None:
            max_len = lengths.max()
        padded = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            length = min(seq.shape[0], max_len)
            padded[i, :length, :] = seq[:length, :]
        return padded

    def encode_protein_batch(self, protein_sequences, max_len=None):
        """Encode and pad a batch of protein sequences."""
        one_hot_seqs = [self.protein_one_hot(seq) for seq in protein_sequences]
        return self.pad_one_hot(one_hot_seqs, max_len=max_len)


# --------------------------
# Full pipeline
# --------------------------
def prepare_data(df, max_len):
    """
    df: dataframe with columns 'Sequence protein' and 'Class'
    max_len: max padded length for protein sequences
    Returns:
        X_padded: (n_samples, max_len, 20)
        y: numeric labels
        label_encoder: sklearn LabelEncoder object
    """
    encoder = GeneEncoder()

    # Protein sequences
    protein_sequences = df["Sequence protein"].astype(str).tolist()

    # Encode & pad
    X_padded = encoder.encode_protein_batch(protein_sequences, max_len=max_len)

    # Convert class labels to numeric
    classes = df["Class"].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(classes)

    return X_padded, y, le
