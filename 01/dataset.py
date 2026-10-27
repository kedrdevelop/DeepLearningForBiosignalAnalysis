'''
MIT-BIH Arrhythmia Database: heartbeat dataset for Case Study 1.

On the first call, load_mitbih() downloads the records from PhysioNet (~100 MB),
cuts one window around each annotated heartbeat and caches the result in
<datadir>/dataset.npz. Later calls only read the cache.

Teacher: run `python dataset.py` once, then give data/dataset.npz to the students. If this
file is in <datadir>, nothing is downloaded. An interrupted download is resumed.

Fast path for students: if mitbih_beats.npz is already in <datadir> (given by the
teacher), nothing is downloaded. An interrupted download is resumed (only the
missing files are fetched).

Adapted from https://github.com/lxysl/mit-bih_ecg_recognition.git
'''
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import wfdb
from wfdb.processing import correct_peaks

# PhysioNet's own server is throttled (~10 KB/s); the same files are on its AWS open-data mirror
MIRROR = 'https://physionet-open.s3.amazonaws.com/mitdb/1.0.0'

FS = 360  # sampling frequency of the MIT-BIH recordings (Hz)

# MIT-BIH beat annotation symbols -> NSVQ classes (Llamedo et al., V and F merged).
# Annotations that do not mark a heartbeat (rhythm changes '+', noise '~', ...) are discarded.
BEAT_TO_CLASS = {
    # N: normal and bundle branch block beats
    'N': 'N', '.': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'B': 'N',
    # S: supraventricular ectopic beats
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V: ventricular ectopic beats (fusion beats merged)
    'V': 'V', 'E': 'V', 'F': 'V', 'f': 'V',
    # Q: paced or unclassifiable beats
    '/': 'Q', 'Q': 'Q',
}

# Inter-patient split proposed by de Chazal et al. (paced records excluded)
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
       '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']

KEYS = ('Xtrain', 'ytrain', 'rtrain', 'Xtest', 'ytest', 'rtest')


def missing_files(wfdb_dir):
    '''Files of the DS1 and DS2 records that are absent or empty (e.g. interrupted download).'''
    files = [f'{r}.{ext}' for r in DS1 + DS2 for ext in ('hea', 'dat', 'atr')]
    return [f for f in files
            if not os.path.exists(os.path.join(wfdb_dir, f)) or os.path.getsize(os.path.join(wfdb_dir, f)) == 0]


def missing_files(wfdb_dir):
    '''Files of the DS1 and DS2 records that are absent or empty (e.g. interrupted download).'''
    files = [f'{r}.{ext}' for r in DS1 + DS2 for ext in ('hea', 'dat', 'atr')]
    return [f for f in files
            if not os.path.exists(os.path.join(wfdb_dir, f)) or os.path.getsize(os.path.join(wfdb_dir, f)) == 0]


def download_records(wfdb_dir, workers=8, retries=3):
    '''
    Download the DS1 and DS2 records (signals + beat annotations) from PhysioNet.

    Only the missing files are downloaded, several at a time, so the function can be
    called again to resume an interrupted download.
    '''
    os.makedirs(wfdb_dir, exist_ok=True)

    def fetch(name):
        # download to a temporary name, then rename: a killed download never leaves a truncated file
        tmp = os.path.join(wfdb_dir, name + '.part')
        urllib.request.urlretrieve(f'{MIRROR}/{name}', tmp)
        os.replace(tmp, os.path.join(wfdb_dir, name))

    for attempt in range(retries):
        todo = missing_files(wfdb_dir)
        if not todo:
            return
        print(f'Downloading {len(todo)} files from PhysioNet (attempt {attempt + 1}/{retries})...')
        with ThreadPoolExecutor(workers) as pool:
            for future in [pool.submit(fetch, f) for f in todo]:
                try:
                    future.result()
                except Exception as e:  # network error: the file is retried at the next attempt
                    print('  failed:', e)
    todo = missing_files(wfdb_dir)
    if todo:
        raise RuntimeError(f'{len(todo)} files could not be downloaded ({todo[:3]}...). '
                           'Check your connection and run again.')


def extract_beats(wfdb_dir, records, left=60, right=300):
    '''
    Cut a window of left+right samples around each annotated heartbeat (lead MLII).

    Returns
    -------
    X: float32 array (n_beats, left+right), ECG amplitude in mV
    y: str array (n_beats,), beat class among N, S, V, Q
    r: str array (n_beats,), name of the record (= patient) the beat comes from
    '''
    X, y, r = [], [], []
    for record in records:
        path = os.path.join(wfdb_dir, record)
        ecg = wfdb.rdrecord(path, channel_names=['MLII']).p_signal[:, 0]
        ann = wfdb.rdann(path, 'atr')

        symbols = np.array(ann.symbol)
        is_beat = np.isin(symbols, list(BEAT_TO_CLASS))
        peaks = correct_peaks(ecg, ann.sample[is_beat], search_radius=60, smooth_window_size=15)
        labels = np.array([BEAT_TO_CLASS[s] for s in symbols[is_beat]])

        # discard beats too close to the borders of the recording
        inside = (peaks - left >= 0) & (peaks + right <= len(ecg))
        for peak, label in zip(peaks[inside], labels[inside]):
            X.append(ecg[peak - left:peak + right])
            y.append(label)
            r.append(record)
    return np.stack(X).astype(np.float32), np.array(y), np.array(r)


def load_mitbih(datadir='data'):
    '''
    Load the heartbeat dataset: train set = records DS1, test set = records DS2.

    Returns Xtrain, ytrain, rtrain, Xtest, ytest, rtest (see extract_beats).
    '''
    cache = os.path.join(datadir, 'dataset.npz')
    if not os.path.exists(cache):
        wfdb_dir = os.path.join(datadir, 'mitdb')
        download_records(wfdb_dir)  # does nothing if all the files are already there
        print('Extracting heartbeats...')
        Xtrain, ytrain, rtrain = extract_beats(wfdb_dir, DS1)
        Xtest, ytest, rtest = extract_beats(wfdb_dir, DS2)
        tmp = cache + '.tmp.npz'
        np.savez_compressed(tmp, Xtrain=Xtrain, ytrain=ytrain, rtrain=rtrain,
                            Xtest=Xtest, ytest=ytest, rtest=rtest)
        os.replace(tmp, cache)
    with np.load(cache) as data:
        return tuple(data[k] for k in KEYS)


if __name__ == '__main__':
    Xtrain, ytrain, rtrain, Xtest, ytest, rtest = load_mitbih('data')
    print(Xtrain.shape, Xtest.shape)
