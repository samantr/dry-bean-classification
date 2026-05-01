from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def resolve_dataset_path(file_path=None):
    """Return a robust absolute path to the dataset.

    Resolution order:
    1) explicit file_path if it exists
    2) project_root/data/Dry_Bean_Dataset.xlsx
    3) first .xlsx file inside project_root/data
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'

    candidates = []
    if file_path:
        user_path = Path(file_path)
        if user_path.is_absolute():
            candidates.append(user_path)
        else:
            candidates.extend([
                Path.cwd() / user_path,
                project_root / user_path,
                data_dir / user_path.name,
            ])

    candidates.append(data_dir / 'Dry_Bean_Dataset.xlsx')

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    excel_files = sorted(data_dir.glob('*.xlsx'))
    if excel_files:
        return excel_files[0].resolve()

    raise FileNotFoundError(
        f'No Excel dataset found. Checked candidates: {[str(c) for c in candidates]}. '
        f'Also searched in: {data_dir}'
    )


def load_and_preprocess_data(file_path=None, test_size=0.2, random_state=42):
    dataset_path = resolve_dataset_path(file_path)
    print(f'Loading dataset from: {dataset_path}', flush=True)

    df = pd.read_excel(dataset_path)

    X = df.drop('Class', axis=1)
    y = df['Class']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, label_encoder
