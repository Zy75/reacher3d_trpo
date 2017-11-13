https://github.com/joschu/modular_rl
https://gist.github.com/joschu/6de0710846dff7230543016fc7639f82

python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=500 --seed=0 --timesteps_per_batch=35000 --env=RoboschoolReacher3d-v1 --outfile=$outdir/Reacher3d

in run_pg.py
line 27 delete: env.monitor.start(mondir, video_callable=None if args.video else VIDEO_NEVER)
add:    env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
add: import roboschool
