import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_ncde_training_history(
    history_clean: pd.DataFrame,
    history_ragged: pd.DataFrame,
    title: str | None = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    for ax, (label, hist) in zip(axes, [('Clean', history_clean),
                                        ('Ragged', history_ragged)]):
        mean_curve = hist.groupby('epoch')['train_loss'].mean()
        p25_curve  = hist.groupby('epoch')['train_loss'].quantile(0.25)
        p75_curve  = hist.groupby('epoch')['train_loss'].quantile(0.75)

        ax.plot(mean_curve.index, mean_curve.values,
                color='C0', linewidth=2, label='Mean')
        ax.fill_between(mean_curve.index, p25_curve, p75_curve,
                        alpha=0.2, color='C0', label='IQR across steps')
        ax.axvline(mean_curve.index[-1], color='grey', linestyle=':',
                   linewidth=0.8, label=f'Final epoch ({mean_curve.index[-1]})')

        # Minimum mean loss.
        best_epoch = mean_curve.idxmin()
        ax.axvline(best_epoch, color='red', linestyle='--',
                   linewidth=0.8, label=f'Min at epoch {best_epoch}')

        ax.set(title=f'NCDENow ({label}): mean training loss ± IQR',
               xlabel='Epoch', ylabel='MSE (annualized pp²)')
        ax.legend(fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=14)
    
    return fig, axes


def plot_terminal_loss(
    history_clean: pd.DataFrame,
    history_ragged: pd.DataFrame,
    title: str | None = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots(figsize=(12, 4))

    for label, hist, color in [('Clean', history_clean, 'C0'),
                                ('Ragged', history_ragged, 'C1')]:
        terminal = (hist.groupby('step')['train_loss'].last()
                        .reset_index()
                        .rename(columns={'train_loss': 'final_loss'}))
        # Map step to nowcast date.
        step_to_date = (hist.groupby('step')['nowcast_date']
                            .first()
                            .reset_index())
        terminal = terminal.merge(step_to_date, on='step')

        ax.plot(terminal['nowcast_date'], terminal['final_loss'],
                color=color, linewidth=1.5, marker='o',
                markersize=3, alpha=0.8, label=f'NCDENow ({label})')

    # Highlight crisis windows.
    ax.axvspan(pd.Timestamp('2008-08-01'), pd.Timestamp('2009-06-01'),
            alpha=0.10, color='grey', label='GFC window')
    ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'),
            alpha=0.10, color='orange', label='COVID window')
    ax.set(title=title, xlabel='Nowcast date',
           ylabel='Final epoch MSE (annualized pp²)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend()

    return fig, ax


def plot_predictions(
    results_dict: dict,
    title: str | None = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    COLORS = {
        'Ridge':                  'C0',
        'Random Forest':          'C1',
        'NCDENow (clean)':        'C2',
        'NCDENow (ragged)':       'C3',
        'NCDENow (clean, fixed)': 'C2',
        'NCDENow (ragged, fixed)':'C3',
    }
    DEFAULT_COLORS = [f'C{i}' for i in range(10)]

    # Highlight crisis windows (GFC and COVID).
    CRISIS_SPANS = [
        (pd.Timestamp('2008-08-01'), pd.Timestamp('2009-06-01'), 'GFC window', 'grey'),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'), 'COVID window', 'orange')
    ]
    
    # Expanding window and fixed split results have different test spans.
    all_dates = pd.concat([
        pd.to_datetime(df['nowcast_date']) for df in results_dict.values()
    ])
    plot_start = all_dates.min()
    plot_end   = all_dates.max()
    
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot a crisis only if it falls within the common time span.
    for span_start, span_end, span_label, span_color in CRISIS_SPANS:
        if span_end >= plot_start and span_start <= plot_end:
            ax.axvspan(span_start, span_end,
                       alpha=0.12, color=span_color,
                       zorder=0, label=span_label)

    # Plot true GDP growth (get once for all data frames).
    first_df = next(iter(results_dict.values()))
    first_df = first_df.copy()
    first_df['nowcast_date'] = pd.to_datetime(first_df['nowcast_date'])
    first_df = first_df.sort_values('nowcast_date')
    ax.plot(first_df['nowcast_date'], first_df['y_true'],
            color='black', linewidth=2.0, zorder=3,
            label='True GDP growth')
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':', zorder=1)

    # Plot model predictions.
    for i, (label, df) in enumerate(results_dict.items()):
        df = df.copy()
        df['nowcast_date'] = pd.to_datetime(df['nowcast_date'])
        df = df.sort_values('nowcast_date')
        color = COLORS.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.plot(df['nowcast_date'], df['y_pred'],
                color=color, linewidth=1.2, linestyle='--',
                alpha=0.85, zorder=2, label=label)

    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.set_ylabel('Annualized QoQ real GDP growth (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis='x', rotation=30)
    ax.legend(loc='lower center', fontsize=10, ncol=3, framealpha=0.9)

    return fig, ax
