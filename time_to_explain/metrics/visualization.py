def read_tabs_plot(files, name, plot_only_og=False):
    tabs = {k: pd.read_csv(v).groupby("sparsity").mean() for k, v in files.items()}
    best_fids = {k: tab["fid_inv_best"].max() for k, tab in tabs.items()}
    aufsc = {k: np.trapz(tab["fid_inv_best"], tab.index) for k, tab in tabs.items()}
    print("Best Fid:", best_fids)
    print("AUFSC:", aufsc)
    for k, tab in tabs.items():
        if k not in ["xtg-og", "attn", "pbone", "pg"] and plot_only_og:
            continue
        plt.plot(tab.index, tab["fid_inv_best"], label=labels[k], marker=markers[k])
    plt.xlabel("Sparsity")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.savefig(f"plots/{name}.png")
    plt.show()
    return tabs, best_fids, aufsc