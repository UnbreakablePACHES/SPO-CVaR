import optuna

from utils.backtester import SPOBacktester
from utils.factories import ModelFactory
from utils.seed_manager import SeedManager
from utils.trainer import SPOTrainer


class SPOHyperTuner:
    """Optuna tuner for SPO rolling backtests."""

    def __init__(
        self,
        df,
        model_type,
        n_assets,
        base_hyperparams,
        model_args=None,
        seed=42,
        n_trials=20,
        trading_days_path=None,
        objective_metric="Sharpe Ratio",
        search_space=None,
        backtest_kwargs=None,
    ):
        self.df = df
        self.model_type = model_type
        self.n_assets = n_assets
        self.base_hyperparams = dict(base_hyperparams)
        self.model_args = dict(model_args or {})
        self.seed = seed
        self.n_trials = int(n_trials)
        self.trading_days_path = trading_days_path
        self.objective_metric = objective_metric
        self.search_space = dict(search_space or {})
        self.backtest_kwargs = dict(backtest_kwargs or {})

    def _suggest_float(self, trial, name, default_low, default_high, log=False):
        bounds = self.search_space.get(name, [default_low, default_high])
        if bounds is None:
            return self.base_hyperparams[name]
        return trial.suggest_float(name, float(bounds[0]), float(bounds[1]), log=log)

    def _suggest_int(self, trial, name, default_low, default_high):
        bounds = self.search_space.get(name, [default_low, default_high])
        if bounds is None:
            return self.base_hyperparams[name]
        return trial.suggest_int(name, int(bounds[0]), int(bounds[1]))

    def _suggest_categorical(self, trial, name, default_choices):
        choices = self.search_space.get(name, default_choices)
        if choices is None:
            return self.base_hyperparams[name]
        return trial.suggest_categorical(name, choices)

    def _trial_hyperparams(self, trial):
        hp = dict(self.base_hyperparams)
        hp["lr"] = self._suggest_float(trial, "lr", 1e-4, 1e-2, log=True)
        hp["epochs"] = self._suggest_int(trial, "epochs", 5, 30)
        hp["window_months"] = self._suggest_categorical(
            trial, "window_months", [6, 12, 18]
        )
        return hp

    def objective(self, trial):
        SeedManager.set_seed(self.seed)
        hp = self._trial_hyperparams(trial)
        model_params = {**hp, **self.model_args, "seed": self.seed}
        opt_model = ModelFactory.get_opt_model(
            self.model_type, n_assets=self.n_assets, **model_params
        )
        backtester = SPOBacktester(
            opt_model=opt_model, trading_days_path=self.trading_days_path
        )

        try:
            backtester.run(
                df=self.df,
                trainer_cls=SPOTrainer,
                window_months=hp["window_months"],
                epochs=hp["epochs"],
                lr=hp["lr"],
                batch_size=hp["batch_size"],
                freq=hp["rebalance_freq"],
                seed=self.seed,
                context_history=hp.get("context_history", 20),
                label_window=int(hp.get("label_window", 21)),
                **self.backtest_kwargs,
            )
            metrics = backtester.evaluate(self.df, fee_rate=hp["fee_rate"])
            return float(metrics[self.objective_metric])
        except Exception as exc:
            print(f"Trial failed: {exc}")
            return -1e9

    def tune(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print("\n" + "=" * 30)
        print("Best SPO hyperparameters:")
        print(study.best_params)
        print(f"Best {self.objective_metric}: {study.best_value:.4f}")
        print("=" * 30)
        return study
