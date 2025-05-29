from django.test import TestCase, Client
from django.urls import reverse
import json
import base64


class TransformViewsTestCase(TestCase):
    """测试变换视图函数"""
    
    def setUp(self):
        self.client = Client()
    
    def test_index_view(self):
        """测试主页视图"""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '变换可视化工具')
    
    def test_fourier_view(self):
        """测试傅里叶变换页面"""
        response = self.client.get(reverse('fourier'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '傅里叶变换')
    
    def test_laplace_view(self):
        """测试拉普拉斯变换页面"""
        response = self.client.get(reverse('laplace'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '拉普拉斯变换')
    
    def test_wavelet_view(self):
        """测试小波变换页面"""
        response = self.client.get(reverse('wavelet'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '小波变换')
    
    def test_hough_view(self):
        """测试霍夫变换页面"""
        response = self.client.get(reverse('hough'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '霍夫变换')
    
    def test_z_transform_view(self):
        """测试Z变换页面"""
        response = self.client.get(reverse('z_transform'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Z变换')
    
    def test_fractional_fourier_view(self):
        """测试分数傅里叶变换页面"""
        response = self.client.get(reverse('fractional_fourier'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '分数傅里叶变换')


class TransformAPITestCase(TestCase):
    """测试变换API接口"""
    
    def setUp(self):
        self.client = Client()
        self.api_url = reverse('api_transform')
    
    def test_api_transform_get_method_not_allowed(self):
        """测试GET方法不被允许"""
        response = self.client.get(self.api_url)
        self.assertEqual(response.status_code, 405)
    
    def test_fourier_transform_api(self):
        """测试傅里叶变换API"""
        data = {
            'transform_type': 'fourier',
            'frequency': 5,
            'sample_rate': 1000
        }
        response = self.client.post(
            self.api_url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertTrue(response_data['success'])
        self.assertIn('image', response_data)
        self.assertIn('info', response_data)
        
        # 验证返回的是有效的base64图像
        try:
            base64.b64decode(response_data['image'])
        except Exception:
            self.fail("返回的图像数据不是有效的base64格式")
    
    def test_laplace_transform_api(self):
        """测试拉普拉斯变换API"""
        data = {
            'transform_type': 'laplace',
            'system_param': 2
        }
        response = self.client.post(
            self.api_url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertTrue(response_data['success'])
        self.assertIn('image', response_data)
    
    def test_wavelet_transform_api(self):
        """测试小波变换API"""
        data = {
            'transform_type': 'wavelet',
            'noise_level': 0.1,
            'wavelet': 'db4'
        }
        response = self.client.post(
            self.api_url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertTrue(response_data['success'])
        self.assertIn('image', response_data)
    
    def test_unknown_transform_type(self):
        """测试未知变换类型"""
        data = {
            'transform_type': 'unknown',
        }
        response = self.client.post(
            self.api_url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        response_data = response.json()
        self.assertIn('error', response_data)
    
    def test_invalid_json(self):
        """测试无效JSON数据"""
        response = self.client.post(
            self.api_url,
            data='invalid json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 500)


class TransformHandlersTestCase(TestCase):
    """测试变换处理函数"""
    
    def test_plot_to_base64_function(self):
        """测试plot_to_base64函数"""
        from .transform_handlers import plot_to_base64
        import matplotlib.pyplot as plt
        
        # 创建一个简单的图
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Test Plot')
        
        # 测试转换函数
        result = plot_to_base64()
        
        # 验证结果是有效的base64字符串
        self.assertIsInstance(result, str)
        try:
            base64.b64decode(result)
        except Exception:
            self.fail("plot_to_base64返回的不是有效的base64格式")
