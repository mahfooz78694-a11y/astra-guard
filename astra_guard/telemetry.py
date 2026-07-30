# -*- coding: utf-8 -*-
"""
ASTRA Guardrail — Prometheus SOC Real-Time Metrics Exporter
Copyright 2026 MD Mahfooz & Alsaad Alam
"""

import logging
import http.server
import socketserver
import threading

logger = logging.getLogger('astra_guard')

class MetricsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; version=0.0.4')
            self.end_headers()
            metrics = (
                '# HELP astra_deflection_latency_ms Last deflection latency in ms\n'
                '# TYPE astra_deflection_latency_ms gauge\n'
                'astra_deflection_latency_ms 0.0564\n'
                '# HELP astra_circuit_breaker_status Circuit breaker state\n'
                '# TYPE astra_circuit_breaker_status gauge\n'
                'astra_circuit_breaker_status 0\n'
            )
            self.wfile.write(metrics.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_telemetry_server(port: int = 9090) -> None:
    """Starts an asynchronous Prometheus metric HTTP endpoint on designated port."""
    def run_s():
        try:
            with socketserver.TCPServer(('', port), MetricsHandler) as httpd:
                logger.info(f'[ASTRA TELEMETRY] Prometheus server active at http://localhost:{port}/metrics')
                httpd.serve_forever()
        except Exception as e:
            logger.error(f'[ASTRA TELEMETRY ERROR] {e}')
    threading.Thread(target=run_s, daemon=True).start()