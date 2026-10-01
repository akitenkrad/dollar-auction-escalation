use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "dollar-auction", about = "Simulation entry point")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一シミュレーションを実行する
    Run,
    /// パラメータスイープを実行する
    Sweep,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run => {
            // TODO: implement run
            // 実行設定を `<output_dir>/config.json` に書き出すこと．
            // `dollar-auction-tools show-experiment-settings --results-dir <output_dir>` で読み出し可能にする．
            println!("TODO: implement run");
        }
        Commands::Sweep => {
            // TODO: implement sweep
            // スイープ設定を `<output_dir>/sweep_config.json` に書き出すこと．
            // `--snapshot-interval N` (N>0) を受け付け，各 run サブディレクトリに
            // `snapshots/step_*.csv` を出力する経路を用意すること．
            // `dollar-auction-tools visualize-sweep` の組み合わせ別グリッドアニメーションが
            // この snapshots を必要とする．
            println!("TODO: implement sweep");
        }
    }
}
