"""Run a longer training budget with resumable checkpoints."""

import argparse
import os

import numpy as np
import torch

from model import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--L", type=int, default=4, help="number of RISs")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--pmax_dbm", type=float, default=15.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_iter", type=int, default=30000)
    p.add_argument("--log_eval_interval", type=int, default=1000)
    p.add_argument("--test_sample_val", type=int, default=400)
    p.add_argument("--test_sample_final", type=int, default=3200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="results_long_training")
    p.add_argument("--resume", default=None,
                   help="checkpoint to warm start from; accepts a plain state_dict or "
                        "a resumable bundle that also carries the optimizer state")
    p.add_argument("--seed", type=int, default=None,
                   help="re-seed after importing trainer_2, which otherwise fixes the seed to 0. "
                        "Varies AP/RIS geometry, model init and channel draws together.")
    p.add_argument("--save_every", type=int, default=2000,
                   help="write a resumable checkpoint every N iterations (0 disables)")
    args = p.parse_args()

    import trainer_2
    from trainer_2 import Trainer

    if args.seed is not None:
        import random
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[INFO] re-seeded to {args.seed} (overrides trainer_2's fixed seed 0)")

    exp = f"M{args.M}_N{args.N}_L{args.L}_K{args.K}_P{args.pmax_dbm}_iter{args.n_iter}"
    if args.seed is not None:
        exp += f"_seed{args.seed}"
    base = os.path.join(args.out_dir, exp, "run0")
    os.makedirs(os.path.join(base, "arrays"), exist_ok=True)

    trainer = Trainer(args.M, args.N, args.L, args.K, args.batch_size,
                      args.pmax_dbm, device=args.device)

    opt_state, start_iter = None, 0
    if args.resume:
        bundle = load_checkpoint(trainer.model, args.resume, trainer.device)
        if isinstance(bundle, dict) and "model" in bundle:
            opt_state = bundle.get("optimizer")
            start_iter = int(bundle.get("iteration", 0))
        print(f"[resume] {args.resume} (iteration {start_iter}, "
              f"optimizer state {'restored' if opt_state else 'NOT available'})")

    trainer.n_iter = args.n_iter
    trainer.log_eval_interval = args.log_eval_interval

    # `Trainer.train` builds its optimizer internally, so inject the restored
    # state and the periodic saving on the first training step instead.
    ckpt_dir = os.path.join(base, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    original_train_batch = trainer.train_batch
    state = {"step": 0, "restored": opt_state is None}

    def train_batch_with_checkpoints():
        if not state["restored"]:
            trainer.opt.load_state_dict(opt_state)
            state["restored"] = True
        out = original_train_batch()
        state["step"] += 1
        if args.save_every and state["step"] % args.save_every == 0:
            torch.save({"model": trainer.model.state_dict(),
                        "optimizer": trainer.opt.state_dict(),
                        "iteration": start_iter + state["step"]},
                       os.path.join(ckpt_dir, "resumable_latest.pt"))
        return out

    trainer.train_batch = train_batch_with_checkpoints
    print(f"[INFO] {exp}: {trainer.n_iter} iterations, validating every "
          f"{trainer.log_eval_interval}, vectorized forward enabled")

    trainer.train(run_id=0, out_dir=base, log_dir=os.path.join(base, "logs"),
                  test_sample_val=args.test_sample_val,
                  test_sample_final=args.test_sample_final)

    torch.save({"model": trainer.model.state_dict(),
                "optimizer": trainer.opt.state_dict(),
                "iteration": start_iter + state["step"]},
               os.path.join(ckpt_dir, "resumable_final.pt"))
    print(f"[done] total iterations including warm start: {start_iter + state['step']}")


if __name__ == "__main__":
    main()
