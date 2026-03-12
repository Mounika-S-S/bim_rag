# src/evaluation/dashboard.py
"""
Simple dashboard for retrieval metrics only
FIXED: No @ symbols in template, proper data handling
"""
import json
import os
from flask import Flask, render_template_string, request
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)


def load_latest_results(project_id="new"):
    path = f"data/processed/{project_id}/evaluation/results/latest_results.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_per_query_results(project_id="new"):
    path = f"data/processed/{project_id}/evaluation/results/per_query_results.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f).get('results', [])
    return []


def load_routing_results(project_id="new"):
    path = f"data/processed/{project_id}/evaluation/results/routing_results.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f).get('results', [])
    return []


@app.route('/')
def dashboard():
    project = request.args.get('project', 'new')
    
    latest = load_latest_results(project)
    per_query = load_per_query_results(project)
    routing = load_routing_results(project)
    
    if not latest:
        return "No evaluation data found. Run evaluator.py first."
    
    stats = latest.get('stats', {})
    overall = stats.get('overall', {})
    by_type = stats.get('by_type', {})
    
    return render_template_string(DASHBOARD_TEMPLATE,
                                 project=project,
                                 total=stats.get('total_queries', 0),
                                 overall=overall,
                                 by_type=by_type,
                                 per_query=per_query,
                                 routing=routing,
                                 datetime=datetime)


DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Retrieval Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        
        /* Header */
        .header { 
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white; 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
            font-size: 0.9em;
            margin-right: 10px;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 5px 0;
        }
        .stat-label {
            color: #64748b;
            font-size: 0.85em;
            text-transform: uppercase;
        }
        .good { color: #059669; }
        .warning { color: #d97706; }
        .bad { color: #dc2626; }
        
        /* Section */
        .section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }
        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        th {
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
            padding: 10px 8px;
            text-align: left;
            border-bottom: 2px solid #cbd5e1;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        tr:hover { background: #f8fafc; }
        
        .type-badge {
            background: #3b82f6;
            color: white;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.8em;
            display: inline-block;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .metric-item {
            background: #f8fafc;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.8em;
            color: #64748b;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .footer {
            text-align: center;
            margin: 20px 0;
            color: #64748b;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 RAG Retrieval Dashboard</h1>
            <p>Project: <strong>{{ project }}</strong> | Total Queries: {{ total }} | {{ datetime.now().strftime('%Y-%m-%d %H:%M') }}</p>
            <span class="badge">Precision, Recall, F1</span>
            <span class="badge">MRR, nDCG</span>
            <span class="badge">Hit Rate</span>
            <span class="badge">Router Accuracy</span>
        </div>
        
        <!-- Summary Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(overall.get('routing_accuracy', 0) * 100) }}%</div>
                <div class="stat-label">Routing Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.3f"|format(overall.get('map', 0)) }}</div>
                <div class="stat-label">MAP</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.3f"|format(overall.get('avg_mrr', 0)) }}</div>
                <div class="stat-label">MRR</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {% if overall.get('avg_hit_rate@5', 0) > 0.7 %}good{% elif overall.get('avg_hit_rate@5', 0) > 0.5 %}warning{% else %}bad{% endif %}">
                    {{ "%.1f"|format(overall.get('avg_hit_rate@5', 0) * 100) }}%
                </div>
                <div class="stat-label">Hit Rate@5</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(overall.get('avg_ndcg@5', 0) * 100) }}%</div>
                <div class="stat-label">NDCG@5</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(overall.get('avg_f1@5', 0) * 100) }}%</div>
                <div class="stat-label">F1@5</div>
            </div>
        </div>
        
        <!-- All Metrics Grid -->
        <div class="section">
            <div class="section-title">📈 All Retrieval Metrics</div>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{{ "%.3f"|format(overall.get('map', 0)) }}</div>
                    <div class="metric-label">MAP</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.3f"|format(overall.get('avg_mrr', 0)) }}</div>
                    <div class="metric-label">MRR</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_hit_rate@1', 0) * 100) }}%</div>
                    <div class="metric-label">Hit@1</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_hit_rate@3', 0) * 100) }}%</div>
                    <div class="metric-label">Hit@3</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_hit_rate@5', 0) * 100) }}%</div>
                    <div class="metric-label">Hit@5</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_ndcg@5', 0) * 100) }}%</div>
                    <div class="metric-label">NDCG@5</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_precision@5', 0) * 100) }}%</div>
                    <div class="metric-label">P@5</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_recall@5', 0) * 100) }}%</div>
                    <div class="metric-label">R@5</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{{ "%.1f"|format(overall.get('avg_f1@5', 0) * 100) }}%</div>
                    <div class="metric-label">F1@5</div>
                </div>
            </div>
        </div>
        
        <!-- Router Accuracy -->
        <div class="section">
            <div class="section-title">🎯 Router Accuracy</div>
            {% if routing %}
            <table style="width: auto;">
                <thead>
                    <tr>
                        <th>Retriever</th>
                        <th>Correct</th>
                        <th>Total</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {% set retriever_stats = {} %}
                    {% for r in routing %}
                        {% set ret = r.expected %}
                        {% if ret in retriever_stats %}
                            {% set _ = retriever_stats[ret].__setitem__('total', retriever_stats[ret].total + 1) %}
                            {% if r.correct %}
                                {% set _ = retriever_stats[ret].__setitem__('correct', retriever_stats[ret].correct + 1) %}
                            {% endif %}
                        {% else %}
                            {% set _ = retriever_stats.update({ret: {'total': 1, 'correct': 1 if r.correct else 0}}) %}
                        {% endif %}
                    {% endfor %}
                    
                    {% for retriever, stats in retriever_stats.items()|sort %}
                    <tr>
                        <td><strong>{{ retriever }}</strong></td>
                        <td>{{ stats.correct }}</td>
                        <td>{{ stats.total }}</td>
                        <td class="{% if (stats.correct/stats.total) > 0.8 %}good{% elif (stats.correct/stats.total) > 0.5 %}warning{% else %}bad{% endif %}">
                            {{ "%.1f"|format((stats.correct/stats.total) * 100) }}%
                        </td>
                    </tr>
                    {% endfor %}
                    
                    <tr style="background: #f0f0f0; font-weight: bold;">
                        <td>OVERALL</td>
                        <td>{{ routing|selectattr('correct')|list|length }}</td>
                        <td>{{ routing|length }}</td>
                        <td>{{ "%.1f"|format((routing|selectattr('correct')|list|length / routing|length) * 100) }}%</td>
                    </tr>
                </tbody>
            </table>
            {% else %}
            <p>No routing data available</p>
            {% endif %}
        </div>
        
        <!-- Performance by Query Type -->
        <div class="section">
            <div class="section-title">📊 Performance by Query Type</div>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Hit@5</th>
                        <th>MRR</th>
                        <th>NDCG@5</th>
                        <th>Precision@5</th>
                        <th>Recall@5</th>
                        <th>F1@5</th>
                        <th>MAP</th>
                    </tr>
                </thead>
                <tbody>
                    {% for qtype, stats in by_type.items()|sort %}
                    <tr>
                        <td><span class="type-badge">{{ qtype }}</span></td>
                        <td>{{ stats.query_count }}</td>
                        <td>{{ "%.1f"|format(stats.get('avg_hit_rate@5', 0) * 100) }}%</td>
                        <td>{{ "%.3f"|format(stats.get('avg_mrr', 0)) }}</td>
                        <td>{{ "%.1f"|format(stats.get('avg_ndcg@5', 0) * 100) }}%</td>
                        <td>{{ "%.1f"|format(stats.get('avg_precision@5', 0) * 100) }}%</td>
                        <td>{{ "%.1f"|format(stats.get('avg_recall@5', 0) * 100) }}%</td>
                        <td>{{ "%.1f"|format(stats.get('avg_f1@5', 0) * 100) }}%</td>
                        <td>{{ "%.3f"|format(stats.get('map', 0)) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- Per-Query Results -->
        <div class="section">
            <div class="section-title">🔍 Per-Query Results ({{ per_query|length }} queries)</div>
            <div style="overflow-x: auto; max-height: 500px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Type</th>
                            <th>Query</th>
                            <th>Hit@5</th>
                            <th>MRR</th>
                            <th>NDCG@5</th>
                            <th>P@5</th>
                            <th>R@5</th>
                            <th>F1@5</th>
                            <th>AP</th>
                            <th>Router</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for q in per_query %}
                        <tr>
                            <td>{{ loop.index }}</td>
                            <td><span class="type-badge">{{ q.query_type }}</span></td>
                            <td style="max-width: 250px;">{{ q.query[:60] }}{% if q.query|length > 60 %}...{% endif %}</td>
                            <td>{{ "%.0f"|format(q.get('hit_rate@5', 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format(q.get('mrr', 0)) }}</td>
                            <td>{{ "%.0f"|format(q.get('ndcg@5', 0) * 100) }}%</td>
                            <td>{{ "%.0f"|format(q.get('precision@5', 0) * 100) }}%</td>
                            <td>{{ "%.0f"|format(q.get('recall@5', 0) * 100) }}%</td>
                            <td>{{ "%.0f"|format(q.get('f1@5', 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format(q.get('ap', 0)) }}</td>
                            <td>
                                {% if q.get('route_correct', False) %}
                                <span style="color: #059669;">✓ {{ q.get('actual_retriever', '') }}</span>
                                {% else %}
                                <span style="color: #dc2626;">✗ {{ q.get('actual_retriever', '') }}<br><small>(exp: {{ q.get('expected_retriever', '') }})</small></span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            Retrieval Metrics: Precision, Recall, F1, MRR, nDCG, Hit Rate, MAP | Router Accuracy
        </div>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)