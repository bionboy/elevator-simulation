from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray, float64 as f64
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

# Formatting libs
from rich import inspect, print
from rich.console import Console
from rich import traceback
from rich.progress import track
# Global console for pretty printing and traceback for better debugging
console = Console()
p = console.print
traceback.install()


# region Constants
MINUTES = 60
ARRIVALS_PER_MINUTE = 6
FLOOR_CNT = 4
FLOOR_NAMES = ['g', '2', '3', '4']
ELEVATOR_CAPACITY = 12
TAKE_THE_STAIRS_THRESHOLD = 12
DOOR_OPEN_TIME = .5

# endregion

# region Classes


@dataclass
class Worker():
    arrival: f64 = field(default_factory=f64)
    # bording_time: f64 = field(default_factory=f64)
    bording_time: f64 = np.inf   # type: ignore
    floor: int = field(default_factory=lambda: np.random.randint(1, 4))
    walked: bool = False

    def takeTheStairs(self):
        assert 0 <= self.floor-1 <= 2, 'Target floor can\'t be ground floor'
        probs = [.5, .33, .1]
        self.walked = probs[self.floor-1] > np.random.rand()   # type: ignore
        return self.walked

    def wait(self) -> f64:
        return self.bording_time - self.arrival   # type: ignore


@dataclass
class Elevator():
    floor: int = 0
    capacity: int = ELEVATOR_CAPACITY
    free: int = ELEVATOR_CAPACITY
    workers: list = field(default_factory=list)
    travel_times = np.array([
        [np.nan, 1.0, 1.5, 1.75],
        [1.0, np.nan, .5, .75],
        [1.5, .5, np.nan, .5],
        [1.75, .5, .25, np.nan],
    ])

    def grounded(self):
        return self.floor == 0

    def holding(self):
        return self.capacity - self.free

    def empty(self):
        return self.free == self.capacity

    def full(self):
        return self.free == 0

    def bord(self, worker: Worker, time: Optional[f64] = None):
        assert not self.full(), 'Oh no, tried to bord a full elevator'

        if time is None:
            time = worker.arrival
        worker.bording_time = time

        self.workers.append(worker)
        self.free -= 1

    def targetFloors(self):
        return np.unique(np.array([w.floor for w in self.workers]))

    def Trip(self):
        assert self.floor == 0, 'must start a trip from the gound floor'
        assert not self.empty(), 'must contain workers to make trip'

        duration = f64(0)
        current = 0
        target = None
        target_floors = self.targetFloors()
        for i in range(len(target_floors)):
            # ? Travel to next desired floor
            target = target_floors[i]
            duration += self.travel_times[current, target]
            # ? Stop and open doors
            duration += .5
            current = target

        # ? Reset elevator
        workers = self.workers
        self.workers = []
        self.free = self.capacity

        # ? return to ground floor
        duration += self.travel_times[target, 0]   # type: ignore
        return duration, workers


# endregion


# region Helper Functions

def GetArrivalTimes() -> ndarray:
    inter_arrival_times = np.random.exponential(scale=.1667, size=(MINUTES*ARRIVALS_PER_MINUTE,))
    arrival_times = inter_arrival_times.cumsum()
    arrival_times = arrival_times[arrival_times < 60]
    return arrival_times   # type: ignore


def ArrivedBetween(start, end, times):
    arrivals: ndarray = times[(start < times) & (times < end)]
    return [Worker(a) for a in arrivals]


def splitList(x, idx):
    return x[:idx], x[idx:]

# endregion


def simulation(trial, display=False):
    arrival_times = GetArrivalTimes()
    elevator = Elevator()
    traveled: List[Worker] = []
    queue: List[Worker] = []
    climbers: List[Worker] = []
    oldtime = time = f64(0)

    def logData(time):
        hist.append([str(trial), time, len(queue), len(traveled), len(climbers)])

    hist = []
    if display:
        console.rule('Simulation')
        p('(time) ->    ariv | elev | que | stair')
    while time < 60:
        time += f64(.5)

        # ? Managing queue
        slots = min(elevator.free, len(queue))
        if slots:
            bording, queue = splitList(queue, slots)
            [elevator.bord(w, time) for w in bording]

        # ? Managing new arrivals
        waiting = ArrivedBetween(oldtime, time, arrival_times)
        arrival_cnt = len(waiting)
        if not elevator.full():
            bording, waiting = splitList(waiting, elevator.free)
            [elevator.bord(w, w.arrival) for w in bording]
        if waiting:
            # ? Handle Stair climbers
            if len(queue) > 12:
                for idx, w in enumerate(waiting):
                    if w.takeTheStairs():
                        climbers.append(waiting.pop(idx))
            # ? Log queue growth
            for w in waiting:
                queue.append(w)
                logData(w.arrival)

        # ? Run the elevator
        holding = elevator.holding()
        trip_time = 0
        if not elevator.empty():
            trip_time, travelers = elevator.Trip()
            traveled += travelers

        # ? Print info
        if display:
            p(f'([red]8:{time:05.2f}[/red])(+{trip_time:4}) -> {arrival_cnt:4} | {holding} | {len(queue):4} | {len(climbers)}')
        # hist.append([time, len(queue)])
        logData(time)

        # ? Advance simulation time
        oldtime = time
        time += trip_time
    return hist, (traveled, climbers, queue)


def main(trials=2, display_runs=False):
    hists = []
    waiting_times: List[f64] = []
    last_borders: List[f64] = []
    climbers: List[Worker] = []
    for trial in track(range(trials), "Running Simulation Trials"):
        hist, (traveled, climbed, queue) = simulation(trial, display=display_runs)

        workers = traveled + queue
        waits = [w.wait() for w in workers if w.bording_time != np.inf]
        waiting_times += waits

        last_bording_time = np.array([w.bording_time for w in traveled]).max()
        last_borders.append(last_bording_time)

        climbers += climbed

        if display_runs:
            console.rule('workers')
            # workers.sort(key=lambda x: x.arrival)
            pct_traveled = (len(traveled) / len(workers)) * 100
            p(f'% Traveled: {pct_traveled:.2f}%')

            console.rule('Avg Wait time')
            # ! ASSUMPTION: Wait times ignore those who never bord
            waits = np.array(waits)
            p(f'Min: {waits.min()}, Avg: {waits.mean()}, Max: {waits.max()}')

            console.rule('Climbers')
            floors, cnts = np.unique(np.array([w.floor for w in climbed]), return_counts=True)
            for floor, cnt in zip(floors, cnts):
                p(f'floor {floor} -> {cnt}', end=', ')
            p()

        hists += hist

    df = DataFrame(hists, columns=['trial', 'time', 'queue', 'traveled', 'climbers'])
    #! Rounding to the minute for seaborn
    df['time'] = df['time'].apply(int, convert_dtype=True)
    df.drop_duplicates(inplace=True)   # type: ignore
    # p(df.info())   # type: ignore

    console.rule('Average Waiting Time')
    p(np.array(waiting_times).mean())

    console.rule('Climbers Per Day')
    floors = np.array([w.floor for w in climbers])
    floors, cnts = np.unique(floors, return_counts=True)
    for floor, cnt in zip(floors, cnts):
        p(f'floor {floor+1} -> {(cnt/trials):6.2f}')

    console.rule('Last Riders Each Day')
    break_cnt = 0
    for idx, t in enumerate(last_borders):
        dec_part = int(str(t-int(t))[2:])
        p(f'Day {idx} -> [red]8:{int(t):02}:{dec_part:02}[/red]', end=' | ')
        break_cnt += 1
        if break_cnt == 5:
            break_cnt = 0
            p()

    console.rule('Average Number of Workers in line at 8:30, 8:45, and 9:00')
    #! Have to use this instead of 60 minutes
    end = df['time'].max()
    p(f'[red]8:30[/red] -> {df[df["time"] == 30]["queue"].mean()}')
    p(f'[red]8:45[/red] -> {df[df["time"] == 45]["queue"].mean()}')
    p(f'[red]9:00[/red] -> {df[df["time"] == end]["queue"].mean()}')

    with console.status('Plotting'):
        lines = ['queue', 'traveled', 'climbers']
        for line in lines:
            sns.lineplot(data=df, x='time', y=line)

        sns.set()
        plt.title(f'Elevator Simulation (Trials={trials})')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Count (humans)')
        plt.legend(labels=lines)
        plt.tight_layout()  # type: ignore
        # plt.show()


if __name__ == '__main__':
    main(trials=10, display_runs=False)
