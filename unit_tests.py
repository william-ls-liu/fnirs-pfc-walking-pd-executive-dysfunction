import unittest
import fnirs_io
import pandas as pd
from processing.process import _transform_data, process_fnirs
from processing.ssc_regression import _find_short, ssc_regression
from processing.average_channels import average_channels
from processing.baseline import baseline_subtraction
import numpy as np
import math


class TestTransformData(unittest.TestCase):
    def setUp(self):
        self.raw = fnirs_io.read_raw("test_data/Turn_511_LongWalk_DT.txt")
        self.raw_data = self.raw['data']
        self.short, self.long, self.events = _transform_data(
            self.raw_data,
            self.raw['metadata'],
            ['Rx1-Tx4', 'Rx2-Tx6'])

    def test_short_channels(self):
        short_test = self.raw_data.filter(regex="Rx1-Tx4|Rx2-Tx6")
        pd.testing.assert_frame_equal(short_test, self.short)

    def test_long_channels(self):
        long_test_regex = "Rx1-Tx1|Rx1-Tx2|Rx1-Tx3|Rx2-Tx5|Rx2-Tx7|Rx2-Tx8"
        long_test = self.raw_data.filter(regex=long_test_regex)
        pd.testing.assert_frame_equal(long_test, self.long)


class TestFindShort(unittest.TestCase):
    def setUp(self):
        self.raw = fnirs_io.read_raw("test_data/Turn_511_LongWalk_DT.txt")
        self.raw_data = self.raw['data']
        self.metadata = self.raw['metadata']
        self.short, self.long, self.events = _transform_data(
            self.raw_data,
            self.metadata,
            ['Rx1-Tx4', 'Rx2-Tx6'])

    def test_find_short(self):
        self.assertEqual(_find_short('Rx1-Tx1 O2Hb', self.short),
                         'Rx1-Tx4 O2Hb')
        self.assertEqual(_find_short('Rx1-Tx1 HHb', self.short),
                         'Rx1-Tx4 HHb')
        self.assertEqual(_find_short('Rx1-Tx2 HHb', self.short),
                         'Rx1-Tx4 HHb')
        self.assertEqual(_find_short('Rx1-Tx3 HHb', self.short),
                         'Rx1-Tx4 HHb')
        self.assertEqual(_find_short('Rx2-Tx5 O2Hb', self.short),
                         'Rx2-Tx6 O2Hb')
        self.assertEqual(_find_short('Rx2-Tx5 HHb', self.short),
                         'Rx2-Tx6 HHb')
        self.assertEqual(_find_short('Rx2-Tx8 O2Hb', self.short),
                         'Rx2-Tx6 O2Hb')
        with self.assertRaises(KeyError):
            _find_short('Rx1-Tx8 o2', self.short)


class TestSSC(unittest.TestCase):
    def setUp(self):
        self.raw = fnirs_io.read_raw("test_data/Turn_511_LongWalk_DT.txt")
        self.raw_data = self.raw['data']
        self.metadata = self.raw['metadata']
        self.short, self.long, self.events = _transform_data(
            self.raw_data,
            self.metadata,
            ['Rx1-Tx4', 'Rx2-Tx6'])

    @staticmethod
    def ssc_test(long_data, short_data):
        short_data_copy = short_data.copy()
        long_data_copy = long_data.copy()
        corrected_df = long_data.copy()
        long_chs = list(long_data.columns)
        for long_ch in long_chs:
            short_ch = _find_short(long_ch, short_data_copy)
            long_array = np.array(long_data_copy[long_ch], dtype='float64')
            short_array = np.array(short_data_copy[short_ch], dtype='float64')

            alpha = (
                np.dot(short_array, long_array) / np.dot(short_array, short_array)
                )
            corrected = long_array - (alpha * short_array)
            corrected_df[long_ch] = corrected

        return corrected_df

    def test_new_ssc(self):
        test_frame = self.ssc_test(self.long, self.short)
        module_frame = ssc_regression(self.long, self.short)
        pd.testing.assert_frame_equal(test_frame, module_frame)


class TestAverageChannels(unittest.TestCase):
    def setUp(self):
        self.raw = fnirs_io.read_raw("test_data/Turn_511_LongWalk_DT.txt")
        self.raw_data = self.raw['data']
        self.metadata = self.raw['metadata']
        self.processed = process_fnirs(self.raw, ['Rx1-Tx4', 'Rx2-Tx6'])

    @staticmethod
    def merge_channels(df):
        df_copy = df.copy()
        dict_for_df = {'Sample number': df_copy['Sample number']}
        right_oxy = ['Rx1-Tx1 O2Hb', 'Rx1-Tx2 O2Hb', 'Rx1-Tx3 O2Hb']
        right_dxy = ['Rx1-Tx1 HHb', 'Rx1-Tx2 HHb', 'Rx1-Tx3 HHb']
        left_oxy = ['Rx2-Tx5 O2Hb', 'Rx2-Tx7 O2Hb', 'Rx2-Tx8 O2Hb']
        left_dxy = ['Rx2-Tx5 HHb', 'Rx2-Tx7 HHb', 'Rx2-Tx8 HHb']

        r_oxy_list = list()
        r_dxy_list = list()
        l_oxy_list = list()
        l_dxy_list = list()
        g_oxy_list = list()
        g_dxy_list = list()

        for ch in list(df_copy.columns):
            if 'Sample number' in ch or 'Event' in ch:
                continue
            if ch in right_oxy:
                r_oxy_list.append(df_copy[ch].tolist())
                g_oxy_list.append(df_copy[ch].tolist())
            elif ch in right_dxy:
                r_dxy_list.append(df_copy[ch].tolist())
                g_dxy_list.append(df_copy[ch].tolist())
            elif ch in left_oxy:
                l_oxy_list.append(df_copy[ch].tolist())
                g_oxy_list.append(df_copy[ch].tolist())
            elif ch in left_dxy:
                l_dxy_list.append(df_copy[ch].tolist())
                g_dxy_list.append(df_copy[ch].tolist())

        dict_for_df['right oxy'] = [math.fsum(tup)/3 for tup in zip(*r_oxy_list)]
        dict_for_df['right dxy'] = [math.fsum(tup)/3 for tup in zip(*r_dxy_list)]
        dict_for_df['left oxy'] = [math.fsum(tup)/3 for tup in zip(*l_oxy_list)]
        dict_for_df['left dxy'] = [math.fsum(tup)/3 for tup in zip(*l_dxy_list)]
        dict_for_df['grand oxy'] = [math.fsum(tup)/6 for tup in zip(*g_oxy_list)]
        dict_for_df['grand dxy'] = [math.fsum(tup)/6 for tup in zip(*g_dxy_list)]
        dict_for_df['Event'] = df_copy['Event']

        return pd.DataFrame(dict_for_df)

    def test_average(self):
        test_frame = self.merge_channels(self.processed)
        test_frame = test_frame.round(6)
        module_frame = average_channels(self.processed)
        module_frame = module_frame.round(6)
        pd.testing.assert_frame_equal(test_frame, module_frame)


class TestBaseline(unittest.TestCase):
    def setUp(self):
        samples = np.arange(0, 24, dtype=np.float64)
        events = np.full(24, np.nan)
        events[2] = 1
        events[6] = 2
        events[20] = 3
        data = np.arange(10, 34)
        corrected_data = data - 13.5
        self.frame = pd.DataFrame(data={'Sample number': samples,
                                        'Data': data,
                                        'Event': events})
        self.events_frame = self.frame[self.frame['Event'].notnull()]
        self.corrected_frame = pd.DataFrame(data={'Data': corrected_data})
        self.test_frame = pd.DataFrame(data={'Data': data})

    def test_baseline(self):
        bad_events = self.events_frame.copy()
        bad_events.drop(index=2, inplace=True)
        og_frame = baseline_subtraction(self.test_frame, self.events_frame)
        pd.testing.assert_frame_equal(self.corrected_frame, og_frame)

        with self.assertRaises(ValueError):
            baseline_subtraction(self.test_frame, bad_events)


unittest.main(verbosity=2)
