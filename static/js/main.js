/**
 * Enhanced Monte Carlo Simulation Platform
 * Advanced visualization and interaction utilities
 */

// ==================== UTILITY FUNCTIONS ====================

const Utils = {
    /**
     * Format number with appropriate precision
     */
    formatNumber: (num, decimals = 4) => {
        if (num === null || num === undefined) return 'N/A';
        if (Math.abs(num) < 0.0001 && num !== 0) {
            return num.toExponential(2);
        }
        return parseFloat(num).toFixed(decimals);
    },

    /**
     * Format currency values
     */
    formatCurrency: (amount) => {
        if (amount >= 1000000) {
            return '$' + (amount / 1000000).toFixed(2) + 'M';
        } else if (amount >= 1000) {
            return '$' + (amount / 1000).toFixed(1) + 'K';
        } else {
            return '$' + amount.toFixed(2);
        }
    },

    /**
     * Format percentage
     */
    formatPercent: (value, decimals = 2) => {
        return (value * 100).toFixed(decimals) + '%';
    },

    /**
     * Show loading indicator
     */
    showLoading: (containerId) => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="flex items-center justify-center py-12">
                    <div class="spinner"></div>
                    <span class="ml-3 text-gray-600">Calculating...</span>
                </div>
            `;
        }
    },

    /**
     * Show error message
     */
    showError: (message, containerId = null) => {
        const errorHtml = `
            <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
                <div class="flex items-center">
                    <svg class="w-6 h-6 text-red-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <div>
                        <p class="font-semibold text-red-800">Error</p>
                        <p class="text-red-700 text-sm">${message}</p>
                    </div>
                </div>
            </div>
        `;

        if (containerId) {
            document.getElementById(containerId).innerHTML = errorHtml;
        } else {
            // Create floating notification
            const notification = document.createElement('div');
            notification.className = 'fixed top-4 right-4 z-50 animate-slide-in';
            notification.innerHTML = errorHtml;
            document.body.appendChild(notification);

            setTimeout(() => notification.remove(), 5000);
        }
    },

    /**
     * Show success message
     */
    showSuccess: (message) => {
        const successHtml = `
            <div class="bg-green-50 border-l-4 border-green-500 p-4 rounded-lg">
                <div class="flex items-center">
                    <svg class="w-6 h-6 text-green-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <div>
                        <p class="font-semibold text-green-800">Success</p>
                        <p class="text-green-700 text-sm">${message}</p>
                    </div>
                </div>
            </div>
        `;

        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 z-50 animate-slide-in';
        notification.innerHTML = successHtml;
        document.body.appendChild(notification);

        setTimeout(() => notification.remove(), 3000);
    },

    /**
     * Download data as CSV
     */
    downloadCSV: (data, filename) => {
        const csv = Utils.convertToCSV(data);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    },

    /**
     * Convert data to CSV format
     */
    convertToCSV: (data) => {
        if (!data || data.length === 0) return '';

        const headers = Object.keys(data[0]);
        const rows = data.map(row => 
            headers.map(header => {
                const value = row[header];
                return typeof value === 'string' ? `"${value}"` : value;
            }).join(',')
        );

        return [headers.join(','), ...rows].join('\n');
    },

    /**
     * Download as JSON
     */
    downloadJSON: (data, filename) => {
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }
};

// ==================== PLOTTING LIBRARY ====================

const PlotlyTheme = {
    layout: {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f9fafb',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            size: 12,
            color: '#374151'
        },
        xaxis: {
            gridcolor: '#e5e7eb',
            zerolinecolor: '#d1d5db'
        },
        yaxis: {
            gridcolor: '#e5e7eb',
            zerolinecolor: '#d1d5db'
        },
        colorway: ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    },
    config: {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    }
};

const Plots = {
    /**
     * Create line plot for convergence analysis
     */
    convergencePlot: (elementId, data, options = {}) => {
        const traces = [];

        // MC estimates
        traces.push({
            x: data.simulations,
            y: data.estimates,
            mode: 'lines+markers',
            name: 'MC Estimate',
            line: { color: '#4f46e5', width: 2 },
            marker: { size: 6 }
        });

        // True value line
        if (data.trueValue) {
            traces.push({
                x: data.simulations,
                y: Array(data.simulations.length).fill(data.trueValue),
                mode: 'lines',
                name: 'True Value',
                line: { color: '#10b981', width: 2, dash: 'dash' }
            });
        }

        // Confidence bands
        if (data.confidenceBands) {
            traces.push({
                x: data.simulations.concat(data.simulations.slice().reverse()),
                y: data.confidenceBands.upper.concat(data.confidenceBands.lower.slice().reverse()),
                fill: 'toself',
                fillcolor: 'rgba(79, 70, 229, 0.2)',
                line: { color: 'transparent' },
                name: '95% CI',
                showlegend: true
            });
        }

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || 'Convergence Analysis',
            xaxis: {
                title: 'Number of Simulations',
                type: 'log',
                ...PlotlyTheme.layout.xaxis
            },
            yaxis: {
                title: options.yLabel || 'Estimate',
                ...PlotlyTheme.layout.yaxis
            },
            hovermode: 'x unified'
        };

        Plotly.newPlot(elementId, traces, layout, PlotlyTheme.config);
    },

    /**
     * Create histogram with distribution overlay
     */
    distributionPlot: (elementId, data, options = {}) => {
        const traces = [];

        // Histogram
        traces.push({
            x: data.values,
            type: 'histogram',
            name: 'Frequency',
            nbinsx: options.bins || 30,
            marker: {
                color: '#4f46e5',
                opacity: 0.7
            }
        });

        // Overlay normal distribution if requested
        if (options.showNormal) {
            const mean = data.values.reduce((a, b) => a + b, 0) / data.values.length;
            const std = Math.sqrt(data.values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.values.length);
            
            const xRange = Array.from({length: 100}, (_, i) => 
                Math.min(...data.values) + i * (Math.max(...data.values) - Math.min(...data.values)) / 99
            );
            
            const normalDist = xRange.map(x => 
                Math.exp(-Math.pow(x - mean, 2) / (2 * std * std)) / (std * Math.sqrt(2 * Math.PI))
            );

            traces.push({
                x: xRange,
                y: normalDist.map(y => y * data.values.length * (Math.max(...data.values) - Math.min(...data.values)) / (options.bins || 30)),
                mode: 'lines',
                name: 'Normal Fit',
                line: { color: '#10b981', width: 2 }
            });
        }

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || 'Distribution',
            xaxis: {
                title: options.xLabel || 'Value',
                ...PlotlyTheme.layout.xaxis
            },
            yaxis: {
                title: 'Frequency',
                ...PlotlyTheme.layout.yaxis
            },
            barmode: 'overlay'
        };

        Plotly.newPlot(elementId, traces, layout, PlotlyTheme.config);
    },

    /**
     * Create path plot for stochastic processes
     */
    pathPlot: (elementId, pathData, options = {}) => {
        const traces = pathData.map((path, idx) => ({
            x: path.map(p => p.time),
            y: path.map(p => p.price),
            mode: 'lines',
            name: `Path ${idx + 1}`,
            line: {
                width: 1.5,
                color: PlotlyTheme.layout.colorway[idx % PlotlyTheme.layout.colorway.length]
            },
            opacity: 0.7
        }));

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || 'Simulated Paths',
            xaxis: {
                title: 'Time',
                ...PlotlyTheme.layout.xaxis
            },
            yaxis: {
                title: options.yLabel || 'Price',
                ...PlotlyTheme.layout.yaxis
            },
            hovermode: 'x unified',
            showlegend: pathData.length <= 10
        };

        Plotly.newPlot(elementId, traces, layout, PlotlyTheme.config);
    },

    /**
     * Create 3D surface plot
     */
    surfacePlot: (elementId, surfaceData, options = {}) => {
        const trace = {
            x: surfaceData.x,
            y: surfaceData.y,
            z: surfaceData.z,
            type: 'surface',
            colorscale: 'Viridis',
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#42f462",
                    project: { z: true }
                }
            }
        };

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || '3D Surface',
            scene: {
                xaxis: { title: options.xLabel || 'X' },
                yaxis: { title: options.yLabel || 'Y' },
                zaxis: { title: options.zLabel || 'Z' }
            },
            autosize: true,
            height: 600
        };

        Plotly.newPlot(elementId, [trace], layout, PlotlyTheme.config);
    },

    /**
     * Create comparison bar chart
     */
    comparisonChart: (elementId, data, options = {}) => {
        const trace = {
            x: data.categories,
            y: data.values,
            type: 'bar',
            marker: {
                color: data.values.map((_, idx) => 
                    PlotlyTheme.layout.colorway[idx % PlotlyTheme.layout.colorway.length]
                )
            },
            text: data.values.map(v => v.toFixed(4)),
            textposition: 'auto'
        };

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || 'Comparison',
            xaxis: {
                title: options.xLabel || '',
                ...PlotlyTheme.layout.xaxis
            },
            yaxis: {
                title: options.yLabel || 'Value',
                ...PlotlyTheme.layout.yaxis
            }
        };

        Plotly.newPlot(elementId, [trace], layout, PlotlyTheme.config);
    },

    /**
     * Create scatter plot with regression
     */
    scatterWithRegression: (elementId, data, options = {}) => {
        const traces = [];

        // Scatter points
        traces.push({
            x: data.x,
            y: data.y,
            mode: 'markers',
            name: 'Data',
            marker: {
                size: 6,
                color: '#4f46e5',
                opacity: 0.6
            }
        });

        // Regression line if provided
        if (data.regression) {
            traces.push({
                x: data.regression.x,
                y: data.regression.y,
                mode: 'lines',
                name: `Fit (R² = ${data.regression.r2.toFixed(3)})`,
                line: {
                    color: '#ef4444',
                    width: 2
                }
            });
        }

        const layout = {
            ...PlotlyTheme.layout,
            title: options.title || 'Scatter Plot',
            xaxis: {
                title: options.xLabel || 'X',
                ...PlotlyTheme.layout.xaxis
            },
            yaxis: {
                title: options.yLabel || 'Y',
                ...PlotlyTheme.layout.yaxis
            }
        };

        Plotly.newPlot(elementId, traces, layout, PlotlyTheme.config);
    }
};

// ==================== RESULTS TABLE BUILDER ====================

const TableBuilder = {
    /**
     * Create results table
     */
    createTable: (containerId, data, options = {}) => {
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = `
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
        `;

        // Headers
        options.headers.forEach(header => {
            html += `<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">${header}</th>`;
        });

        html += `
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
        `;

        // Rows
        data.forEach((row, idx) => {
            html += `<tr class="${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}">`;
            
            options.columns.forEach(col => {
                const value = row[col];
                const formatted = typeof value === 'number' ? Utils.formatNumber(value) : value;
                html += `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${formatted}</td>`;
            });
            
            html += `</tr>`;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Create metrics card grid
     */
    createMetricsGrid: (containerId, metrics) => {
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = '<div class="grid grid-cols-2 md:grid-cols-4 gap-4">';

        metrics.forEach(metric => {
            const colorClass = metric.color || 'indigo';
            html += `
                <div class="bg-white rounded-lg shadow p-4 border-l-4 border-${colorClass}-500">
                    <div class="text-sm text-gray-600">${metric.label}</div>
                    <div class="text-2xl font-bold text-${colorClass}-600 mt-1">${metric.value}</div>
                    ${metric.subtitle ? `<div class="text-xs text-gray-500 mt-1">${metric.subtitle}</div>` : ''}
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }
};

// ==================== EXPORT TO GLOBAL SCOPE ====================

window.Utils = Utils;
window.Plots = Plots;
window.TableBuilder = TableBuilder;
window.PlotlyTheme = PlotlyTheme;

// ==================== PAGE INTERACTIONS ====================

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Add input validation
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);

            if (!isNaN(min) && value < min) {
                this.classList.add('border-red-500');
            } else if (!isNaN(max) && value > max) {
                this.classList.add('border-red-500');
            } else {
                this.classList.remove('border-red-500');
            }
        });
    });
});