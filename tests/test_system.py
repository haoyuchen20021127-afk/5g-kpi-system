"""Run: python -m unittest discover -s tests -v"""
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
from core.config import PROJECT_ROOT
from core import prediction as pr, anomaly as ad
from core.detectors import compare_score, finite_quantile, fit_detector


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.meta, cls.spec = pr.read_test_data()
        cls.model = pr.load_model(pr.file_signature(), cls.spec['X_sha256'])
        cls.stamps = {k: ad.signature(k) for k in ad.KPIS}
        cls.ref = ad.custom_reference('0_0', 'test_5w-6w', 840, cls.stamps)

    def test_assets_intact(self):
        for row in json.loads((PROJECT_ROOT/'evidence/asset_manifest.json').read_text(encoding='utf-8')):
            self.assertEqual(hashlib.sha256((PROJECT_ROOT/row['path']).read_bytes()).hexdigest(), row['sha256'])

    def test_prediction_regression(self):
        expected = pd.read_csv(PROJECT_ROOT/'assets/prediction/models/LSTM/test_predictions_seed_42.csv')
        got = pr.predict_history(self.model, self.x)
        self.assertEqual(got.shape, (144, 4))
        # Archived predictions were computed on CUDA, this app runs on CPU.
        np.testing.assert_allclose(got, expected[[f'predicted__{k}' for k in pr.TARGETS]], atol=1e-5, rtol=1e-5)

    def test_prediction_edit(self):
        changed = self.x[:1].copy()
        changed[:, -3:, 0] += .3
        self.assertGreater(float(np.max(np.abs(pr.predict_history(self.model, changed)-pr.predict_history(self.model,self.x[:1])))), 1e-5)

    def test_prediction_invalid(self):
        for value in [np.zeros((1,23,4)), np.full((1,24,4),np.nan), np.zeros((0,24,4))]:
            with self.assertRaises(ValueError): pr.predict_history(self.model,value)

    def test_custom_no_change(self):
        for method in ad.METHODS:
            table, old, new = ad.compare_custom(self.ref,ad.custom_editor_frame(self.ref),method,self.stamps)
            np.testing.assert_array_equal(table['原始分数'],table['当前分数'])
            np.testing.assert_array_equal(old.flag,new.flag)

    def test_custom_independent_kpis(self):
        frame = ad.custom_editor_frame(self.ref)
        frame.loc[0,'当前输入值'] = 2.
        table, old, new = ad.compare_custom(self.ref,frame,'isolation_forest',self.stamps)
        self.assertEqual(old.flag.iloc[0],0)
        self.assertEqual(new.flag.iloc[0],1)
        np.testing.assert_array_equal(old.score.iloc[1:],new.score.iloc[1:])
        self.assertEqual(ad.record_summary(new).flag.iloc[0],1)

    def test_invalid_custom_input(self):
        for value in [-1, np.nan, np.inf, 1e100]:
            frame = ad.custom_editor_frame(self.ref)
            frame.loc[0,'当前输入值'] = value
            with self.assertRaises(ValueError): ad.validate_custom(frame,self.ref)

    def test_record_or_with_missing(self):
        base = pd.DataFrame({'cell':['a']*4,'window':['test']*4,'time':[1]*4,'kpi':ad.KPIS})
        for flags, expected in [([0,0,0,0],0), ([1,-1,0,0],1), ([0,-1,0,0],-1)]:
            base['flag']=flags
            self.assertEqual(ad.record_summary(base).flag.iloc[0],expected)

    def test_guard_boundary(self):
        flags, _ = compare_score([0.,1e-6,2e-6,np.nan],0.,'one_class_svm')
        np.testing.assert_array_equal(flags,[0,0,1,-1])
        self.assertEqual(finite_quantile(np.arange(100)),95.)

    def test_offline_detector_fit(self):
        fit = np.linspace(-2,2,120)
        cal = np.linspace(-2,2,60)
        for method in ad.METHODS:
            detector = fit_detector(method,fit,cal)
            self.assertEqual(detector.status,'ok')
            self.assertTrue(np.isfinite(detector.score([0.,10.])).all())


class PageTests(unittest.TestCase):
    def test_prediction_modes(self):
        from streamlit.testing.v1 import AppTest
        page = AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=60).run()
        self.assertEqual(len(page.exception),0)
        page.button(key='ip_apply_shift').click().run()
        page.button(key='ip_predict_custom').click().run()
        self.assertEqual(len(page.exception),0)
        pair=page.session_state['ip_scenario']['prediction']
        self.assertGreater(float(np.max(np.abs(pair[0]-pair[1]))),1e-5)
        page.button(key='ip_restore').click().run()
        self.assertNotIn('ip_scenario',page.session_state)
        page.radio(key='ip_mode').set_value('历史测试查询').run()
        page.button(key='ip_predict_history').click().run()
        self.assertEqual(len(page.exception),0)
        self.assertEqual(page.session_state['ip_full'].shape,(144,4))
        page.selectbox(key='ip_filter_rule').set_value('误差最大的前 N 条').run()
        self.assertEqual(len(page.exception),0)
        page.radio(key='ip_mode').set_value('自定义输入预测').run()
        self.assertEqual(len(page.exception),0)

    def test_anomaly_modes(self):
        from streamlit.testing.v1 import AppTest
        page = AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=60).run()
        page.switch_page('app_pages/anomaly.py').run()
        self.assertEqual(len(page.exception),0)
        page.button(key='ad_run').click().run()
        self.assertEqual(len(page.exception),0)
        self.assertGreater(len(page.metric),0)
        page.radio(key='ad_mode').set_value('自定义数据检测').run()
        page.button(key='ac_run').click().run()
        self.assertEqual(len(page.exception),0)
        self.assertEqual(len(page.metric),2)


if __name__ == '__main__':
    unittest.main()
