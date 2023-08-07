import pickle
import numpy as np
import torch
from torch import nn


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


with open("X_test.pkl", "rb") as file:
    X_test = pickle.load(file)
with open("y_test_unnormalized.pkl", "rb") as file:
    y_test = pickle.load(file)
with open("y_test.pkl", "rb") as file:
    y_test_normalized = pickle.load(file)
with open("y_baseline_test.pkl", "rb") as file:
    y_baseline_test = pickle.load(file)
with open("qid_test.pkl", "rb") as file:
    qid_test = pickle.load(file)

device = "cuda"

model = NN().to(device)
model.load_state_dict(torch.load("nn.0.007179192898166753.0.04731562372526994.pt"))
# model = torch.compile(model)

qid_test_unique = np.unique(qid_test)
print(qid_test_unique[:10])

avg_rel_top1 = 0
avg_rel_top2 = 0
avg_rel_top3 = 0
avg_rel_err_true = 0

acc_top1 = 0
acc_top2 = 0
acc_top3 = 0

import time
X_test = torch.from_numpy(X_test).to(device)

with torch.no_grad():
    for i, test_id in enumerate(qid_test_unique):
        time_start = time.time()
        scores = model.forward(X_test[qid_test == test_id]).cpu().numpy().squeeze()
        time_end = time.time()
        if i < 20:
            print(
                f"test_id: {test_id}, time: {time_end - time_start}, len: {X_test[qid_test == test_id].shape[0]}"
            )
    
        y_pred_arr = y_test[qid_test == test_id][np.argsort(scores)[::-1]]
        y_pred_top1 = y_pred_arr[:1].min()
        y_pred_top2 = y_pred_arr[:2].min()
        y_pred_top3 = y_pred_arr[:5].min()

        y_pred = y_pred_arr[y_pred_arr != 1e6][0]
        assert y_pred != 1e6
        y_baseline = y_baseline_test[qid_test == test_id][0]
        y_true = y_test[qid_test == test_id].min()

        acc_top1 += y_pred_top1 == y_true
        acc_top2 += y_pred_top2 == y_true
        acc_top3 += y_pred_top3 == y_true

        avg_rel_top1 += (y_pred_top1 - y_baseline) / y_baseline
        avg_rel_top2 += (y_pred_top2 - y_baseline) / y_baseline
        avg_rel_top3 += (y_pred_top3 - y_baseline) / y_baseline
        avg_rel_err_true += (y_true - y_baseline) / y_baseline

        # print(
        #     f"test_id: {test_id}, y_pred: {y_pred}, y_true: {y_true} y_baseline: {y_baseline}\n",
        #     avg_rel_err / (i + 1) * 100,
        #     avg_rel_err_baseline / (i + 1) * 100,
        #     avg_rel_err_baseline_2 / (i + 1) * 100,
        #     avg_rel_err_baseline_3 / (i + 1) * 100,
        #     avg_rel_err_baseline_true / (i + 1) * 100,
        #     # max_rel_err * 100,
        #     # acc_top1 / (i + 1) * 100,
        #     # acc_top2 / (i + 1) * 100,
        #     # acc_top3 / (i + 1) * 100,
        # )

    print("acc_top1", acc_top1 / len(qid_test_unique) * 100)
    print("acc_top2", acc_top2 / len(qid_test_unique) * 100)
    print("acc_top3", acc_top3 / len(qid_test_unique) * 100)
    print("avg_rel_top1", avg_rel_top1 / len(qid_test_unique) * 100)
    print("avg_rel_top2", avg_rel_top2 / len(qid_test_unique) * 100)
    print("avg_rel_top3", avg_rel_top3 / len(qid_test_unique) * 100)
    print("avg_rel_err_true", avg_rel_err_true / len(qid_test_unique) * 100)


def get_loss(X_loss, y_loss):
    batch_size = 4096
    mse_loss_sum = 0
    mae_loss_sum = 0
    model.eval()

    with torch.no_grad():
        for i in range(0, X_loss.shape[0], batch_size):
            y_pred = model(X_loss[i : i + batch_size]).to(device)
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


mse_loss, mae_loss = get_loss(X_test, y_test_normalized)
print(f"rmse_loss={np.sqrt(mse_loss)} mae_loss={mae_loss}")
