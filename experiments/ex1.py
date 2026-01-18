#import
#taki config 
#dataloader
#data -> aug
#aug -> train
# --- użycie ---

from torchvision import transforms
from data_manager.dataloaders import DataLoaderBuilder
from seed_utils import SEED, set_global_seed

set_global_seed(SEED)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

builder = DataLoaderBuilder(
    csv_path="",
    seed=SEED,
    n_splits=5,
    batch_size=128,
    num_workers=4,
)

train_loader, val_loader, test_loader = builder.make_loaders(
    transform_train=transform_train,
    transform_eval=transform_eval,
)

def _print_leakage_check(train_df, val_df, test_df, group_col="patient_id"):
    tr = set(train_df[group_col].unique())
    va = set(val_df[group_col].unique())
    te = set(test_df[group_col].unique())

    inter_tr_va = len(tr & va)
    inter_tr_te = len(tr & te)
    inter_va_te = len(va & te)

    print("\n[Leakage check - shared patient_id between splits]")
    print(f"  train ∩ val : {inter_tr_va}")
    print(f"  train ∩ test: {inter_tr_te}")
    print(f"  val   ∩ test: {inter_va_te}")

    if inter_tr_va + inter_tr_te + inter_va_te == 0:
        print("  ✅ OK: no patient overlap")
    else:
        print("  ❌ WARNING: patient leakage detected!")
def _test_loader(loader, name="loader"):
    batch = next(iter(loader))
    x, y = batch
    print(f"\n[Smoke test: {name}]")
    print(f"  x: shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item():.3f} max={x.max().item():.3f}")
    print(f"  y: shape={tuple(y.shape)} dtype={y.dtype} unique={y.unique().cpu().numpy()[:10]}")


print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
train_df, val_df, test_df = builder.make_split()
_print_leakage_check(train_df, val_df, test_df)
_test_loader(train_loader, "train_loader")
_test_loader(val_loader,   "val_loader")
_test_loader(test_loader,  "test_loader")
