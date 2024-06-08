import matplotlib.pyplot as plt
import pandas as pd
import torch


def train(
        model, optimizer, criterion, train_loader,
        device: str, epoch_index: int, save_name: str = None
        ) -> tuple[float, float]:

    model.train()
    train_loss = 0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    if save_name is not None:
        torch.save(model.state_dict(), f"{save_name}_{epoch_index}.pt")

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)

    return train_loss, train_accuracy


def test(model, criterion, test_loader, device: str) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy


def plot_models_optimizers(
        df: pd.DataFrame, df_name: str,
        x_axis: str, y_axis: str,
        x_unit: str = None,
        y_unit: str = None
        ) -> None:

    optimizer_names = df["optimizer_name"].unique()
    model_names = df["model_name"].unique()

    model_styles = {
        "Binary": {"linestyle": "--", "marker": "o"},
        "Classic": {"linestyle": "-", "marker": "x"}
        }

    optimizer_colors = {
        "Adam": "red",
        "AdaMax": "blue",
        "AdaDelta": "green"
    }

    plt.figure(figsize=(12, 8))

    for optimizer_name in optimizer_names:
        for model_name in model_names:
            df_filtered = df[
                (df["optimizer_name"] == optimizer_name) &
                (df["model_name"] == model_name)
                ]

            if not df_filtered.empty:
                style = model_styles.get(
                    model_name,
                    {"linestyle": "-", "marker": ""}
                    )

                color = optimizer_colors.get(optimizer_name, "black")
                plt.plot(
                    df_filtered[x_axis],
                    df_filtered[y_axis],
                    label=f"{model_name} ({optimizer_name})",
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    color=color
                    )

    title = f"{df_name}\n{y_axis} on Models "
    title += f"with Different Optimizers Over {x_axis}"
    plt.title(title)

    if x_unit is None:
        plt.xlabel(x_axis.title())
    else:
        plt.xlabel(f"{x_axis.title()} {x_unit}")

    if y_unit is None:
        plt.ylabel(y_axis.title())
    else:
        plt.ylabel(f"{y_axis.title()} {y_unit}")

    plt.legend(
        title="Model (Optimizer)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
        )

    plt.grid(True)
    plt.tight_layout()
    plt.show()
