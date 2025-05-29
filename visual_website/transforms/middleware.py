import time
import logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class PerformanceMonitoringMiddleware:
    """性能监控中间件"""
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        duration = time.time() - start_time
        
        # 记录慢请求（超过2秒）
        if duration > 2.0:
            logger.warning(
                f"慢请求检测: {request.path} 耗时 {duration:.2f}秒"
            )
        
        # 为API请求添加性能头
        if request.path.startswith('/api/'):
            response['X-Response-Time'] = f"{duration:.3f}s"
            
        return response


class TransformErrorHandlingMiddleware:
    """变换错误处理中间件"""
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
            return response
        except Exception as e:
            logger.error(f"未处理的异常: {str(e)}", exc_info=True)
            
            # 如果是API请求，返回JSON错误响应
            if request.path.startswith('/api/'):
                return JsonResponse({
                    'error': '服务器内部错误，请稍后重试',
                    'success': False
                }, status=500)
            
            # 否则重新抛出异常让Django处理
            raise

    def process_exception(self, request, exception):
        """处理视图中的异常"""
        logger.error(f"视图异常: {str(exception)}", exc_info=True)
        
        if request.path.startswith('/api/'):
            return JsonResponse({
                'error': '处理请求时发生错误',
                'success': False
            }, status=500)
        
        return None  # 让Django的默认异常处理器处理
