import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(path='data.csv'):
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['x1']), float(row['x2'])])
            y.append(int(row['label']))
    return np.array(X), np.array(y, dtype=float)


def load_weights(path='model_weights.npz'):
    d = np.load(path)
    return d['W1'], d['b1'], d['W2'], d['b2']


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0)

def forward(X, weights):
    W1, b1, W2, b2 = weights
    b1 = b1.reshape(1, -1)
    b2 = b2.reshape(1, -1)
    rez1 = X @ W1.T + b1
    rel = relu(rez1)

    rez2 = rel @ W2.T + b2
    rez = sigmoid(rez2)

    return rez.reshape(-1), (rez1, rel, rez2)

def bce_loss(y_hat, y):
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1 - eps)

    return -((y * (np.log(y_hat))) + (1 - y) * (np.log(1 - y_hat)))


def compute_gradients(X, y, weights):
    W1, b1, W2, b2 = weights
    N = X.shape[0]
    y_hat, (z1, a1, _) = forward(X, weights)
    d2 = (y_hat - y).reshape(-1, 1)
    grad_W2 = (d2[:, :, np.newaxis] * a1[:, np.newaxis, :])
    grad_b2 = d2
    relu_grad_z1 = (z1 > 0).astype(float)
    d1 = (d2 @ W2) * relu_grad_z1
    grad_W1 = (d1[:, :, np.newaxis] * X[:, np.newaxis, :])
    grad_b1 = d1
    
    return grad_W1, grad_b1, grad_W2, grad_b2

def gradient_check(X, y, weights, eps=1e-5):
    W1, b1, W2, b2 = weights
    sizes = [W1.size, b1.size, W2.size, b2.size]
    shapes = [W1.shape, b1.shape, W2.shape, b2.shape]

    par = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])
    P_s = par.size
    N = X.shape[0]
    Eps = np.eye(P_s) * eps
    par_plus = par + Eps
    par_minus = par - Eps

    def unsp(p):
        split = np.split(p, np.cumsum(sizes)[:-1])
        return (
            split[0].reshape(W1.shape),
            split[1].reshape(b1.shape),
            split[2].reshape(W2.shape),
            split[3].reshape(b2.shape)
        )

    L_plus = []
    L_minus = []
    for i in range(P_s):
        y_hat_p, _ = forward(X, unsp(par_plus[i]))
        y_hat_m, _ = forward(X, unsp(par_minus[i]))

        L_plus.append(bce_loss(y_hat_p, y))
        L_minus.append(bce_loss(y_hat_m, y))
    L_plus = np.array(L_plus)
    L_minus = np.array(L_minus)
    num_grads = (L_plus - L_minus) / (2 * eps)
    split = np.split(num_grads, np.cumsum(sizes)[:-1], axis=0)

    num_grad_W1 = split[0].transpose(1,0).reshape(N, *W1.shape)
    num_grad_b1 = split[1].transpose(1,0).reshape(N, *b1.shape)
    num_grad_W2 = split[2].transpose(1,0).reshape(N, *W2.shape)
    num_grad_b2 = split[3].transpose(1,0).reshape(N, *b2.shape)
    grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(X, y, weights)

    def check_parameters(num, comp, name):
        abs_diff = np.max(np.abs(num - comp))
        rel_diff = np.max(np.abs(num - comp) / (np.abs(comp) + 1e-9))
        passed = (abs_diff < 1e-4) or (rel_diff < 1e-5)
        return {
            'passed': passed,
            'max_abs_diff': abs_diff,
            'max_rel_diff': rel_diff
        }

    return {
        'W1': check_parameters(num_grad_W1, grad_W1, 'W1'),
        'b1': check_parameters(num_grad_b1, grad_b1, 'b1'),
        'W2': check_parameters(num_grad_W2, grad_W2, 'W2'),
        'b2': check_parameters(num_grad_b2, grad_b2, 'b2'),
    }


def input_gradient(x, y_true, weights):
    W1, b1, W2, b2 = weights

    y_true = y_true.reshape(-1, 1)
    y_hat, (z1, a1, z2) = forward(x, weights)
    d2 = y_hat - y_true
    d1 = (d2 @ W2) * relu_grad(z1)
    inp_grad = d1 @ W1

    return inp_grad


def pgd_attack(X, y, weights, lr = 0.05, steps = 200):
    N = X.shape[0]
    dlts = np.zeros_like(X)
    y_hat, _ = forward(X, weights)
    prd = (y_hat > 0.5).astype(int).reshape(-1)
    correct_mask = (prd == y)
    success = np.zeros(N, dtype=bool)

    for i in range(N):
        if not correct_mask[i]:
            continue
        x = X[i].reshape(1, -1)
        y_true = np.array([[y[i]]])
        d = np.zeros_like(x)

        for step in range(steps):
            grad = input_gradient(x + d, y_true, weights)
            grad = grad / (np.linalg.norm(grad) + 1e-9)
            d += lr * grad
            x_new = x + d
            
            y_pred, _ = forward(x_new, weights)
            pred_t = (y_pred > 0.5).astype(int)
            if pred_t != y_true:
                success[i] = True
                break
        dlts[i] = d

    return dlts, success, correct_mask



def plot_decision_boundary(X, y, weights, deltas, success, correct_mask,
                           save_path='adversarial_plot.png'):
    x_min, x_max = X[:,0].min()-0.3, X[:,0].max()+0.3
    y_min, y_max = X[:,1].min()-0.3, X[:,1].max()+0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz, _ = forward(grid, weights)
    zz = zz.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, title, show_adv in zip(axes,
            ['Start + decision boundary',
             'Adversarial examples'],
            [False, True]):

        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1],
                    colors=['#aec6e8','#f4a97a'], alpha=0.35)
        ax.contour(xx, yy, zz, levels=[0.5],
                   colors='#333333', linewidths=1.2)

        ax.scatter(X[y==0, 0], X[y==0, 1], c='#3578b5', s=14,
                   alpha=0.6, label='Class 0', zorder=3)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='#d6604d', s=14,
                   alpha=0.6, label='Class 1', zorder=3)

        if show_adv:
            adv_idx = np.where(success)[0]
            X_adv = X[adv_idx] + deltas[adv_idx]
            norms = np.linalg.norm(deltas[adv_idx], axis=1)

            sc = ax.scatter(X_adv[:, 0], X_adv[:, 1],
                            c=norms, cmap='YlOrRd',
                            s=5, zorder=30, edgecolors='k',
                            linewidths=0.3, label='Adversarial')
            plt.colorbar(sc, ax=ax, label='‖δ‖₂')

            for j in adv_idx[:60]:
                ax.annotate('', xy=X[j]+deltas[j], xytext=X[j],
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=0.5, alpha=0.5))
            ax.set_title(f'{title}\n'
                         f'{success.sum()} successfull attacks '
                         f'{correct_mask.sum()} correct predictions\n'
                         f'Median value ‖δ‖₂ = {np.median(norms):.3f}')
        else:
            ax.set_title(title)

        ax.legend(fontsize=8, markerscale=1.5)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")



if __name__ == '__main__':
    print("=" * 60)
    print("Loading weights")
    print("=" * 60)
    X, y = load_data('data.csv')
    weights = load_weights('model_weights.npz')
    W1, b1, W2, b2 = weights
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"W1: {W1.shape}, b1: {b1.shape}, W2: {W2.shape}, b2: {b2.shape}")

    print("\n" + "=" * 60)
    print("Verifying forward pass")
    print("=" * 60)
    y_hat, _ = forward(X, weights)
    ref = np.load('reference_predictions.npy')
    
    max_diff = np.abs(y_hat - ref).max()
    print(f"Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, "ERROR: bad forward pass!"
    print("Forward pass is ok (< 1e-5)")

    acc = ((y_hat > 0.5) == y.astype(bool)).mean()
    print(f"Dataset accuracy: {acc:.4f}")

    print("\n" + "=" * 60)
    print("Gradient check")
    print("=" * 60)
    idx = np.random.choice(len(X), 50, replace=False)
    gc_results = gradient_check(X[idx], y[idx], list(weights))

    all_passed = True
    for name, res in gc_results.items():
        status = 'ok' if res['passed'] else 'error'
        print(f"  {status} {name:3s}  max_abs_diff={res['max_abs_diff']:.2e}"
              f"  max_rel_diff={res['max_rel_diff']:.2e}"
              f"  {'PASS' if res['passed'] else 'FAIL'}")
        if not res['passed']:
            all_passed = False
    print("Gradients verified" if all_passed
          else "Error in gradients!")

    print("\n" + "=" * 60)
    print("Adversarial examples")
    print("=" * 60)
    deltas, success, correct_mask = pgd_attack(X, y, weights,
                                                lr=0.05, steps=300)

    norms = np.linalg.norm(deltas[success], axis=1)
    print(f"Correct predictions: {correct_mask.sum()}")
    print(f"Successfull attacks: {success.sum()}")
    print(f"‖δ‖₂ — min: {norms.min():.4f}")
    print(f"‖δ‖₂ — median: {np.median(norms):.4f}")
    print(f"‖δ‖₂ — max: {norms.max():.4f}")

    plot_decision_boundary(X, y, weights, deltas, success, correct_mask)