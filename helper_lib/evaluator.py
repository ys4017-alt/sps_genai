import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for test_images, test_labels in data_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            test_loss += criterion(test_outputs, test_labels).item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
            
    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * test_correct / test_total
    print(f"Accuracy of the network on the {test_total} test images: {accuracy:.2f}%")
    return avg_loss, accuracy
    # TODO: calculate average loss and accuracy on the test dataset.