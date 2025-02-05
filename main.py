# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_utils import get_random_balanced_mnist, get_transform_source
from umap_utils import learn_umap_and_rotate
from intermediate_domains import compute_intermediate_domains_barycenter
from training_utils import train_epoch, evaluate, self_training_step
from model import Classifier

def run_experiment(angle_target, dataset_name):
    print(f"\n===== EXPERIMENT: {dataset_name} (target rotation = {angle_target}° applied in R^32) =====")
    
    # a) Load a balanced sample of 4000 source images
    X_source, y_source = get_random_balanced_mnist(num_samples=4000, transform=get_transform_source())
    
    # b) Learn UMAP in 32D on the source and obtain the target by applying a rotation
    X_source_umap, X_target_umap = learn_umap_and_rotate(X_source, y_source, n_components=32, angle_deg=angle_target)
    
    # c) Compute intermediate domains using the Wasserstein barycenter method
    intermediate_features = compute_intermediate_domains_barycenter(
        X_source_umap, X_target_umap, alphas=[0.25, 0.50, 0.75], reg=5e-3)
    
    # d) Construct the 5 domains:
    #    Domain 0: Source, Domains 1-3: intermediate, Domain 4: Target
    domains = []
    domain0 = TensorDataset(torch.tensor(X_source_umap, dtype=torch.float32),
                            torch.tensor(y_source, dtype=torch.long))
    domains.append(domain0)
    for feat in intermediate_features:
        domains.append(TensorDataset(torch.tensor(feat, dtype=torch.float32),
                                     torch.tensor(y_source, dtype=torch.long)))
    domain4 = TensorDataset(torch.tensor(X_target_umap, dtype=torch.float32),
                            torch.tensor(y_source, dtype=torch.long))
    domains.append(domain4)
    
    # e) Create DataLoaders (batch_size = 1024)
    loaders = [DataLoader(domain, batch_size=1024, shuffle=True) for domain in domains]
    target_loader = DataLoader(domain4, batch_size=1024, shuffle=False)
    
    # f) Initialize the classifier, optimizer, and loss function
    classifier = Classifier(input_dim=32, hidden_dim=128, num_classes=10)
    optimizer = optim.SGD(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # g) Step 1: Initial training on the source domain (50 epochs)
    print("Initial training on the source domain...")
    for epoch in range(50):
        loss = train_epoch(classifier, loaders[0], optimizer, criterion)
        acc = evaluate(classifier, loaders[0])
        print(f"  Epoch {epoch+1}/50 - Loss: {loss:.4f} - Source Accuracy: {acc:.4f}")
    
    # h) Step 2: Gradual Self-Training on the intermediate domains (domains 1 to 3)
    for i in range(1, 4):
        print(f"\nGradual Self-Training on intermediate domain {i}...")
        classifier = self_training_step(classifier, loaders[i], optimizer, criterion, num_epochs=50)
    
    # i) Final evaluation on the target domain
    final_acc = evaluate(classifier, target_loader)
    print(f"\nFinal Accuracy on target domain ({dataset_name}): {final_acc:.4f}")
    return final_acc

if __name__ == "__main__":
    acc_45 = run_experiment(angle_target=45, dataset_name="Rotated MNIST 45°")
    acc_60 = run_experiment(angle_target=60, dataset_name="Rotated MNIST 60°")
    
    print("\n===== SUMMARY =====")
    print(f"Final Accuracy (Rotated MNIST 45°): {acc_45:.4f}")
    print(f"Final Accuracy (Rotated MNIST 60°): {acc_60:.4f}")
