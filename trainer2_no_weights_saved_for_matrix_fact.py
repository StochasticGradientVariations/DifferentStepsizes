import numpy as np
import time
import math

import numpy.linalg as la
import matplotlib.pyplot as plt
import os
STORE_WS_EVERY = int(os.environ.get("STORE_WS_EVERY", "0"))  # 0 => μην αποθηκεύεις καθόλου


print_every = 30


class Trainer:
    """
    Base class for experiments with logistic regression. Provides methods
    for running optimization methods, saving the logs and plotting the results.
    """
    def __init__(self, grad_func, loss_func, fmin=0, reg_param=0, prox_type="unconstrained", isVerbose=False, t_max=np.inf, it_max=np.inf, output_size=500, tolerance=0):
        if math.isinf(t_max) and math.isinf(it_max):
            # if t_max is np.inf and it_max is np.inf:

            it_max = 100
            print('The number of iterations is set to 100.')
        self.grad_func = grad_func
        self.loss_func = loss_func
        self.t_max = t_max
        self.it_max = it_max
        self.output_size = output_size
        self.first_run = True
        self.tolerance = tolerance
        self.losses = None
        self.isVerbose = isVerbose
        self.prox_type = prox_type
        self.reg_param = reg_param
        # —–– new fields for tracking progress —––––
        self.loss_hist = []  # tracks the loss value f(w) at each step
        self.grad_norm_hist = []  # stores the gradient norm ∥∇f(w)∥ to monitor convergence
        self.lr_hist = []  # keeps a record of the learning rate used at each iteration

        self.fmin = fmin

    def run(self, w0):
        # — αρχικοποίηση λιστών/μετρητών —
        if self.first_run:
            self.init_run(w0)
        else:
            self.ts = list(self.ts)
            self.its = list(self.its)
            if getattr(self, 'ws', None) is not None:
                self.ws = list(self.ws)

        self.first_run = False

        # — κύριος βρόχος —
        while (self.it < self.it_max) and (time.time() - self.t_start < self.t_max):
            # 1) gradient στο x_k και έλεγχος ανοχής
            sufficiently_big_gradient = self.compute_grad()  # self.grad = ∇f(self.w)
            if not sufficiently_big_gradient:
                break

            # 2) εκτίμηση βήματος & βήμα: x_{k+1}
            self.estimate_stepsize()
            self.w = self.step()

            # 3) gradient στο ΝΕΟ σημείο x_{k+1} για σωστό logging
            self.grad = self.grad_func(self.w)

            # 4) προαιρετικό print προόδου
            if self.isVerbose and (self.it % print_every == 0) and (self.it > 0):
                print(self.it, la.norm(self.grad), self.loss_func(self.w) - self.fmin)

            # 5) logging σε ΚΑΘΕ iteration
            self.save_checkpoint()

        self.ts = np.array(self.ts)
        self.its = np.array(self.its)
        if self.ws is not None and len(self.ws) > 0:
            try:
                self.ws = np.stack(self.ws, axis=0)  # (num_snaps, d) σε float32 αν έκανες downcast
            except Exception:
                self.ws = np.array(self.ws, dtype=np.float32)
        else:
            self.ws = None

    def compute_grad(self):
        self.grad = self.grad_func(self.w)
        return la.norm(self.grad) > self.tolerance
        
    def estimate_stepsize(self):
        pass
        
    def step(self):
        pass

    # def init_run(self, w0):
    #     self.d = len(w0)
    #     self.w = w0.copy()
    #     self.ws = [w0.copy()]
    #     self.its = [0]
    #     self.ts = [0]
    #     self.it = 0
    #     self.t = 0
    #     self.t_start = time.time()
    #     # --- ΑΡΧΙΚΗ ΚΑΤΑΓΡΑΦΗ (k=0) ---
    #     self.grad = self.grad_func(self.w)
    #     self.loss_hist.append(self.loss_func(self.w))
    #     self.grad_norm_hist.append(la.norm(self.grad))
    #     self.lr_hist.append(getattr(self, 'lr', None))  # π.χ. lr0 αν έχει οριστεί στο optimizer
    def init_run(self, w0):
        self.d = len(w0)
        self.w = w0.copy()

        # --- snapshots των weights ανά N iterates (ή καθόλου) ---
        self.ws = [] if STORE_WS_EVERY > 0 else None
        if self.ws is not None:
            # Προαιρετικό: downcast για μνήμη
            self.ws.append(self.w.astype(np.float32, copy=False))

        self.its = [0]
        self.ts = [0]
        self.it = 0
        self.t = 0
        self.t_start = time.time()

        # --- ΑΡΧΙΚΗ ΚΑΤΑΓΡΑΦΗ (k=0) ---
        self.grad = self.grad_func(self.w)
        self.loss_hist.append(self.loss_func(self.w))
        self.grad_norm_hist.append(la.norm(self.grad))
        self.lr_hist.append(getattr(self, 'lr', None))

    # def save_checkpoint(self, first_iterations=10):
    #     self.it += 1
    #     self.t = time.time() - self.t_start
    #     self.time_progress = int((self.output_size - first_iterations) * self.t / self.t_max)
    #     self.iterations_progress = int((self.output_size - first_iterations) * (self.it / self.it_max))
    #     if (max(self.time_progress, self.iterations_progress) > self.max_progress) or (self.it <= first_iterations):
    #         self.update_logs()
    #     self.max_progress = max(self.time_progress, self.iterations_progress)
    def save_checkpoint(self, first_iterations=10):
        # increment iteration counter
        self.it += 1
        # elapsed time
        self.t = time.time() - self.t_start
        # log on EVERY iteration
        self.update_logs()

    # def update_logs(self):
    #     # self.ws.append(self.w.copy())
    #     # self.ts.append(self.t)
    #     # self.its.append(self.it)
    #     # # —–– logging history after each iteration —––––
    #     # self.loss_hist.append(self.loss_func(self.w))  # log current loss value
    #     # self.grad_norm_hist.append(la.norm(self.grad))  # record gradient norm to track optimization progress
    #     # self.lr_hist.append(getattr(self, 'lr', None))  # save the learning rate used (if defined)
    #     # record current parameter state and time info
    #     self.ws.append(self.w.copy())    # save current weights
    #     self.ts.append(self.t)           # save current timestamp or t-value
    #     self.its.append(self.it)         # log current iteration count
    #
    #     # —–– track optimization history —––––
    #     # 1) loss value at current step
    #     self.loss_hist.append(self.loss_func(self.w))
    #
    #     # 2) gradient norm, only if gradient is already available
    #     if hasattr(self, 'grad'):
    #         self.grad_norm_hist.append(la.norm(self.grad))  # log gradient norm
    #     else:
    #         self.grad_norm_hist.append(None)  # store None initially if grad hasn't been computed yet
    #
    #     # 3) learning rate, if defined
    #     self.lr_hist.append(getattr(self, 'lr', None))  # grab current learning rate or None
    def update_logs(self):
        # --- snapshots βαρών μόνο όταν ζητηθεί & αραιά ---
        if self.ws is not None and (self.it % STORE_WS_EVERY == 0):
            self.ws.append(self.w.astype(np.float32, copy=False))

        # χρόνος & iteration
        self.ts.append(self.t)
        self.its.append(self.it)

        # ιστορικά scalar μετρικών
        self.loss_hist.append(self.loss_func(self.w))
        if hasattr(self, 'grad'):
            self.grad_norm_hist.append(la.norm(self.grad))
        else:
            self.grad_norm_hist.append(None)
        self.lr_hist.append(getattr(self, 'lr', None))

    def compute_loss_on_iterates(self):
        if self.ws is None or len(self.ws) == 0:
            # δεν έχουμε snapshots => χρησιμοποίησε τα ήδη καταγεγραμμένα losses
            self.losses = np.array(self.loss_hist, dtype=float)
            return
        self.losses = np.array([self.loss_func(w) for w in self.ws])

    def plot_losses(self, label='', marker=',', f_star=None, markevery=None):
        if self.losses is None:
            self.compute_loss_on_iterates()
        if f_star is None:
            f_star = np.min(self.losses)
        if markevery is None:
            markevery = max(1, len(self.losses) // 20)
        plt.plot(self.its, self.losses - f_star, label=label, marker=marker, markevery=markevery)