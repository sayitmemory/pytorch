import torch
import pickle
import numpy as np
from torch import nn

np.random.seed(0)
np.set_printoptions(edgeitems=30, linewidth=100000, threshold=np.inf, suppress=True)


def show_feature(v):
    np.set_printoptions(suppress=True)
    print("kernel_category", v[0])
    print("num_of_loops", v[1])
    print("op_bag", v[2:58])
    print("size_hints", v[58:60])
    for i in range(10):
        print(f"reads[{i}]", v[60 + i * 17 : 60 + i * 17 + 17])
    for i in range(5):
        print(f"writes[{i}]", v[230 + i * 17 : 230 + i * 17 + 17])
    print("XBLOCK", v[315])
    print("YBLOCK", v[316])
    print("RBLOCK", v[317])
    print("num_warps", v[318])
    print("num_stages", v[319])
    # print("num_regs", v[319])
    # print("num_spills", v[320])
    # print("num_shared", v[321])
    print("xnumel", v[320])
    print("ynumel", v[321])
    print("rnumel", v[322])


with open("X_train.pkl", "rb") as file:
    X_train = pickle.load(file)
with open("y_train.pkl", "rb") as file:
    y_train = pickle.load(file)
with open("qid_train.pkl", "rb") as file:
    qid_train = pickle.load(file)
with open("X_test.pkl", "rb") as file:
    X_test = pickle.load(file)
with open("y_test.pkl", "rb") as file:
    y_test = pickle.load(file)
with open("qid_test.pkl", "rb") as file:
    qid_test = pickle.load(file)

device = "cuda"


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_category_embedding = torch.nn.Embedding(
            num_embeddings=3, embedding_dim=32
        )
        self.num_of_loops_embedding = torch.nn.Embedding(
            num_embeddings=10, embedding_dim=32
        )

        self.hidden_dim = [
            32 + 32 + 8 * 56 + (323 - 2 - 56),
            8192,
            4096,
            2048,
            1024,
            512,
            256,
            128,
            64,
            32,
            1,
        ]
        self.num_layers = len(self.hidden_dim) - 1

        # self.op_bag_ln = nn.Linear(55, 32)
        self.op_bag_ln = nn.ModuleList([nn.Linear(1, 8) for i in range(56)])

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
                for i in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim[i + 1]) for i in range(self.num_layers - 1)]
        )

        torch.nn.init.xavier_normal_(self.kernel_category_embedding.weight)
        torch.nn.init.xavier_normal_(self.num_of_loops_embedding.weight)
        for layer in list(self.op_bag_ln) + list(self.layers):
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.cat(
            [
                self.kernel_category_embedding(x[:, 0].long()),
                self.num_of_loops_embedding(x[:, 1].long()),
                torch.cat(
                    [self.op_bag_ln[i](x[:, 2 + i].unsqueeze(1)) for i in range(56)],
                    dim=1,
                ),
                x[:, 58:],
            ],
            dim=1,
        )
        for norm, layer in zip(self.norms, self.layers[:-1]):
            x = torch.nn.functional.leaky_relu(norm(layer(x)))
        x = torch.sigmoid(self.layers[-1](x))
        return x


model = NN().to(device)
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def get_loss(X_loss, y_loss):
    batch_size = 4096
    mse_loss_sum = 0
    mae_loss_sum = 0
    model.eval()

    with torch.no_grad():
        for i in range(0, X_loss.shape[0], batch_size):
            y_pred = model(torch.from_numpy(X_loss[i : i + batch_size]).to(device))
            mse_loss = torch.nn.functional.mse_loss(
                y_pred.squeeze(),
                torch.from_numpy(y_loss[i : i + batch_size]).to(device),
            )
            mae_loss = torch.nn.functional.l1_loss(
                y_pred.squeeze(),
                torch.from_numpy(y_loss[i : i + batch_size]).to(device),
            )

            mse_loss_sum += mse_loss.item() * y_pred.shape[0]
            mae_loss_sum += mae_loss.item() * y_pred.shape[0]
        torch.cuda.empty_cache()

    return mse_loss_sum / X_loss.shape[0], mae_loss_sum / X_loss.shape[0]


print(model.hidden_dim)
print(lr)

best_mse_loss = 1e9
for epoch in range(10000):
    model.train()
    batch_size = 4096
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    for i in range(0, X_train.shape[0], batch_size):
        optimizer.zero_grad()
        X_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        y_pred = model(torch.from_numpy(X_batch).to(device))
        loss = torch.nn.functional.mse_loss(
            y_pred.squeeze(), torch.from_numpy(y_batch).to(device)
        )
        loss.backward()
        optimizer.step()
        # print(f"{i}/{X_train.shape[0]}", loss.item())

    mse_loss, mae_loss = get_loss(X_train, y_train)
    print(
        f"Train: epoch={epoch} rmse_loss={np.sqrt(mse_loss)} mae_loss={mae_loss}",
        end=" ||| ",
    )

    mse_loss, mae_loss = get_loss(X_test, y_test)
    print(f"Test: epoch={epoch} rmse_loss={np.sqrt(mse_loss)} mae_loss={mae_loss}")

    if mse_loss < best_mse_loss:
        best_mse_loss = mse_loss
        if np.sqrt(mse_loss) < 0.95:
            torch.save(
                model.state_dict(),
                "nn." + str(np.sqrt(mse_loss)) + "." + str(epoch) + ".pt",
            )
