import copy
import sys
import enum
import traceback
import numpy as np
import os
import os.path
import sunpy.io
import h5py
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm


base_path = Path('/mnt/f/GRIS')
kmeans_output_dir = base_path / 'K-Means'
input_files = [
    base_path / '25Apr25ARM2-003.fits',
    base_path / '25Apr25ARM2-004.fits'
]


class Status(enum.Enum):
    Requesting_work = 0
    Work_assigned = 1
    Work_done = 2
    Work_failure = 3


def do_work(num_clusters):
    global shared_array_all_data
    global shared_array_framerows
    global mn
    global sd
    global weights

    try:

        model = KMeans(
            n_clusters=num_clusters,
            max_iter=10000,
            tol=1e-6
        )

        model.fit(shared_array_framerows)

        fout = h5py.File(
            '{}/out_SV_{}.h5'.format(kmeans_output_dir, num_clusters), 'w'
        )

        fout['cluster_centers_'] = model.cluster_centers_
        fout['labels_'] = model.labels_
        fout['inertia_'] = model.inertia_
        fout['n_iter_'] = model.n_iter_

        rps = np.zeros((num_clusters, shared_array_all_data.shape[1]), dtype=np.float64)

        for i in range(num_clusters):
            a = np.where(model.labels_ == i)
            rps[i] = np.mean(shared_array_all_data[a], axis=0)

        fout['rps'] = rps

        fout.close()

        return Status.Work_done
    except Exception:
        sys.stdout.write('Failed for {}\n'.format(num_clusters))
        exc = traceback.format_exc()
        sys.stdout.write(exc)
        return Status.Work_failure


def plot_inertia():

    base_path = Path(kmeans_output_dir)

    k = range(2, 200, 1)

    inertia = list()

    for k_value in k:
        f = h5py.File(
            base_path / 'out_SV_{}.h5'.format(
                k_value
            )
        )

        inertia.append(f['inertia_'][()])
        f.close()

    inertia = np.array(inertia)

    diff_inertia = inertia[:-1]  - inertia[1:]

    plt.close('all')
    plt.clf()
    plt.cla()

    fig, axs = plt.subplots(1, 1, figsize=(5.845, 4.135,))

    axs.plot(k, inertia / 1e5, color='#364f6B')

    axs.set_xlabel('Number of Clusters')

    axs.set_ylabel(r'$\sigma_{k}\;*\;1e5$')

    axs.grid()

    axs.axvline(x=100, linestyle='--')

    # axs.set_xticks([0, 20, 30, 40, 60, 80, 100])

    # axs.set_xticklabels([0, 20, 30, 40, 60, 80, 100])

    ax2 = axs.twinx()

    ax2.plot(k[1:], diff_inertia / 1e5, color='#3fC1C9')

    ax2.set_ylabel(r'$\sigma_{k} - \sigma_{k+1}\;*\;1e5$')

    axs.yaxis.label.set_color('#364f6B')

    ax2.yaxis.label.set_color('#3fC1C9')

    axs.tick_params(axis='y', colors='#364f6B')

    ax2.tick_params(axis='y', colors='#3fC1C9')

    fig.tight_layout()

    fig.savefig(base_path / 'KMeansInertia.pdf', format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


if __name__ == '__main__':
    plot_inertia()


'''
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_path = Path('/mnt/f/GRIS')
    kmeans_output_dir = base_path / 'K-Means'
    input_files = [
        base_path / '25Apr25ARM2-003.fits',
        base_path / '25Apr25ARM2-004.fits'
    ]

    total_rows = 0
    total_cols = 0

    for input_file in input_files:
        data, header = sunpy.io.read_file(input_file)[0]

        if len(data.shape) == 5:
            data = np.transpose(data, axes=(0, 2, 3, 1, 4))

            total_rows += data.shape[0] * data.shape[1] * data.shape[2]

            total_cols = data.shape[4]
        else:
            data = np.transpose(data, axes=(1, 2, 0, 3))

            total_rows += data.shape[0] * data.shape[1]

            total_cols = data.shape[3]

    if shm_comm.rank == 0:
        itemsize = MPI.DOUBLE.Get_size()
        nbytes = total_rows * total_cols * itemsize
    else:
        nbytes = 0
        itemsize = MPI.DOUBLE.Get_size()

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=shm_comm)

    win2 = MPI.Win.Allocate_shared(nbytes, itemsize, comm=shm_comm)

    buf, _ = win.Shared_query(0)

    buf2, _ = win.Shared_query(0)

    shared_array_framerows = np.ndarray(buffer=buf, dtype='d', shape=(total_rows, total_cols))

    shared_array_all_data = np.ndarray(buffer=buf2, dtype='d', shape=(total_rows, total_cols))

    if shm_comm.rank == 0:

        framerows = None

        for input_file in input_files:
            data, header = sunpy.io.read_file(input_file)[0]

            if len(data.shape) == 5:
                data = np.transpose(data, axes=(0, 2, 3, 1, 4))

                data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
            else:
                data = np.transpose(data, axes=(1, 2, 0, 3))

                data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

            if framerows is None:
                framerows = data[:, 3] / data[:, 0]

            else:

                framerows = np.concatenate([framerows, data[:, 3] / data[:, 0]])

        shared_array_all_data[:] = framerows

        mn = np.mean(framerows, axis=0)
        sd = np.std(framerows, axis=0)
        framerows = (framerows - mn) / sd
        weights = np.ones(framerows.shape[1])
        weights[:] = 0.5 / (framerows.shape[1] - 200)
        weights[50:250] = 0.5 / 200
        framerows *= weights[np.newaxis]

        shared_array_framerows[:] = framerows

    comm.Barrier()

    if rank == 0:
        status = MPI.Status()
        waiting_queue = set()
        running_queue = set()
        finished_queue = set()
        failure_queue = set()

        for i in range(2, 200, 1):
            waiting_queue.add(i)

        filepath = '{}/status_job.h5'.format(kmeans_output_dir)
        if os.path.exists(filepath):
            mode = 'r+'
        else:
            mode = 'w'

        try:
            f = h5py.File(filepath, mode)
        except Exception:
            f = h5py.File(filepath, 'w')

        if 'finished' in list(f.keys()):
            finished = list(f['finished'][()])
        else:
            finished = list()

        for index in finished:
            waiting_queue.discard(index)

        t = tqdm(total=len(waiting_queue))

        for worker in range(1, size):
            if len(waiting_queue) == 0:
                break
            item = waiting_queue.pop()
            work_type = {
                'job': 'work',
                'item': item
            }
            comm.send(work_type, dest=worker, tag=1)
            running_queue.add(item)

        while len(running_queue) != 0 or len(waiting_queue) != 0:
            try:
                status_dict = comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=2,
                    status=status
                )
            except Exception:
                sys.exit(1)

            sender = status.Get_source()
            jobstatus = status_dict['status']
            item = status_dict['item']
            running_queue.discard(item)
            if jobstatus == Status.Work_done:
                finished_queue.add(item)
                if 'finished' in list(f.keys()):
                    del f['finished']
                finished.append(item)
                f['finished'] = finished
            else:
                failure_queue.add(item)

            t.update(1)

            if len(waiting_queue) != 0:
                new_item = waiting_queue.pop()
                work_type = {
                    'job': 'work',
                    'item': new_item
                }
                comm.send(work_type, dest=sender, tag=1)
                running_queue.add(new_item)

        f.close()

        for worker in range(1, size):
            work_type = {
                'job': 'stopwork'
            }
            comm.send(work_type, dest=worker, tag=1)

    if rank > 0:

        while 1:
            work_type = comm.recv(source=0, tag=1)

            if work_type['job'] != 'work':
                break

            item = work_type['item']

            status = do_work(item)

            comm.send({'status': status, 'item': item}, dest=0, tag=2)
'''