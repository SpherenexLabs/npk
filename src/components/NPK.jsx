import React, { useState, useEffect, useMemo } from "react";
import { initializeApp, getApps, getApp } from "firebase/app";
import { getDatabase, ref, onValue, off } from "firebase/database";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

/**
 * WaterDetectionDashboard + Client-side KNN suggestions + Real-time Graphs
 * - Adds a lightweight KNN (k=3) recommender that reads current sensors and suggests actions.
 * - Includes real-time graphs for trending sensor data
 * - No external ML libs needed. You can expand/replace the seed dataset below with your historical data.
 */

// ==== Firebase config (yours) ====
const firebaseConfig = {
    apiKey: "AIzaSyB9ererNsNonAzH0zQo_GS79XPOyCoMxr4",
    authDomain: "waterdtection.firebaseapp.com",
    databaseURL: "https://waterdtection-default-rtdb.firebaseio.com",
    projectId: "waterdtection",
    storageBucket: "waterdtection.firebasestorage.app",
    messagingSenderId: "690886375729",
    appId: "1:690886375729:web:172c3a47dda6585e4e1810",
    measurementId: "G-TXF33Y6XY0",
};
const APP_NAME = "water-detection-dashboard-app";

/* ===========================
   1) KNN: Feature engineering
   =========================== */

/**
 * Choose features that influence water quality and dosing:
 * - ph, tds_ppm, turbidity_ntu, temp, ec, water_pct, n, p, k
 * Feel free to adjust weights/ranges as your domain knowledge grows.
 */

// Reasonable operating ranges for min-max normalization (tune as needed)
const FEATURE_RANGES = {
    ph: [0, 14],
    tds_ppm: [0, 2000],       // ppm
    turbidity_ntu: [0, 100],  // NTU
    temp: [-5, 60],           // °C
    ec: [0, 5000],            // µS/cm (example)
    water_pct: [0, 100],      // %
    n: [0, 500],              // arbitrary units or mg/L from your sensor scale
    p: [0, 500],
    k: [0, 500],
};

// Optional per-feature weights if you want to emphasize certain metrics
const FEATURE_WEIGHTS = {
    ph: 1.2,
    tds_ppm: 1.0,
    turbidity_ntu: 1.0,
    temp: 0.6,
    ec: 0.8,
    water_pct: 0.4,
    n: 0.9,
    p: 0.9,
    k: 0.9,
};

// Select feature order once to keep things consistent
const FEATURE_KEYS = Object.keys(FEATURE_RANGES);

/**
 * Normalize a value v from [min,max] -> [0,1]
 */
function norm(v, [min, max]) {
    if (v === undefined || v === null || isNaN(Number(v))) return 0;
    const clamped = Math.max(min, Math.min(max, Number(v)));
    return (clamped - min) / (max - min || 1);
}

/**
 * Vectorize a sample using the agreed feature order and normalization.
 */
function vectorize(sample) {
    return FEATURE_KEYS.map((k) => {
        const nval = norm(sample[k], FEATURE_RANGES[k]);
        const w = FEATURE_WEIGHTS[k] ?? 1;
        return nval * w;
    });
}

/**
 * Squared Euclidean distance between two vectors
 */
function dist2(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/* ===========================
   2) Seed dataset for KNN
   ===========================

   label: A short action class
   suggestion: Human-readable recommendation
   You should gradually replace/augment this with your own labeled dataset
   collected from past situations and outcomes.
*/
const KNN_DATASET = [
    {
        x: { ph: 6.0, tds_ppm: 300, turbidity_ntu: 1, temp: 25, ec: 1200, water_pct: 65, n: 20, p: 10, k: 25 },
        label: "OK",
        suggestion: "Parameters are within acceptable ranges. Maintain current dosing and monitoring.",
    },
    {
        x: { ph: 5.2, tds_ppm: 250, turbidity_ntu: 1, temp: 24, ec: 1000, water_pct: 70, n: 10, p: 8, k: 12 },
        label: "Add Base",
        suggestion: "pH is slightly acidic. Add a mild base or reduce acidic inputs to raise pH toward 6.5–7.5.",
    },
    {
        x: { ph: 8.9, tds_ppm: 350, turbidity_ntu: 1, temp: 27, ec: 1100, water_pct: 60, n: 15, p: 12, k: 20 },
        label: "Add Acid",
        suggestion: "pH is slightly alkaline. Add a mild acid to bring pH closer to 6.5–7.5.",
    },
    {
        x: { ph: 6.7, tds_ppm: 1100, turbidity_ntu: 2, temp: 28, ec: 2000, water_pct: 55, n: 5, p: 3, k: 10 },
        label: "Dilute",
        suggestion: "High TDS/EC detected. Dilute with fresh water or reduce nutrient dosing.",
    },
    {
        x: { ph: 6.8, tds_ppm: 150, turbidity_ntu: 8, temp: 26, ec: 500, water_pct: 70, n: 18, p: 10, k: 20 },
        label: "Filter/Clean",
        suggestion: "Elevated turbidity. Check filters, lines, and consider settling/filtration.",
    },
    {
        x: { ph: 6.5, tds_ppm: 220, turbidity_ntu: 1, temp: 35, ec: 800, water_pct: 40, n: 25, p: 15, k: 30 },
        label: "Cool/Refill",
        suggestion: "High temperature and low water level. Improve cooling and refill the tank.",
    },
    {
        x: { ph: 6.6, tds_ppm: 180, turbidity_ntu: 1, temp: 22, ec: 700, water_pct: 95, n: 5, p: 4, k: 6 },
        label: "Balance NPK",
        suggestion: "Low NPK levels. Consider balanced nutrient dosing based on crop stage.",
    },
    {
        x: { ph: 5.8, tds_ppm: 600, turbidity_ntu: 2, temp: 23, ec: 1400, water_pct: 85, n: 60, p: 20, k: 40 },
        label: "Reduce Nutrients",
        suggestion: "NPK/EC trending high. Reduce nutrient concentration gradually and monitor.",
    },
    {
        x: { ph: 7.2, tds_ppm: 50, turbidity_ntu: 0.5, temp: 20, ec: 150, water_pct: 80, n: 0, p: 0, k: 0 },
        label: "Dose Nutrients",
        suggestion: "Very low TDS/EC. Add base nutrients to reach optimal conductivity.",
    },
    {
        x: { ph: 7.0, tds_ppm: 400, turbidity_ntu: 1, temp: 24, ec: 900, water_pct: 20, n: 18, p: 12, k: 22 },
        label: "Refill Tank",
        suggestion: "Low water level detected. Refill reservoir to avoid pump cavitation and swings.",
    },
];

/**
 * Run KNN for a given sample
 * @param {*} sample - current sensor reading
 * @param {*} k - neighbors
 * @returns { neighbors, majorityLabel, reasons }
 */
function knnSuggest(sample, k = 3) {
    const v = vectorize(sample);
    const scored = KNN_DATASET.map((row) => ({
        row,
        d2: dist2(v, vectorize(row.x)),
    })).sort((a, b) => a.d2 - b.d2);

    const neighbors = scored.slice(0, k);
    // Majority vote
    const counts = new Map();
    for (const n of neighbors) {
        counts.set(n.row.label, (counts.get(n.row.label) || 0) + 1);
    }
    let majorityLabel = null;
    let max = -1;
    for (const [label, c] of counts) {
        if (c > max) {
            max = c;
            majorityLabel = label;
        }
    }

    // Aggregate suggestion text from neighbors with the majority label
    const reasons = neighbors
        .filter((n) => n.row.label === majorityLabel)
        .map((n) => n.row.suggestion);

    // Fallback: if tie, take first neighbor's suggestion
    const suggestionText =
        reasons[0] ||
        neighbors[0]?.row?.suggestion ||
        "No suggestion available. Consider adding more training samples.";

    return { neighbors, majorityLabel, suggestionText };
}

/* ===========================
   3) Chart Utilities
   =========================== */

// Keep last N data points for charts
const MAX_DATA_POINTS = 50;

// Custom tooltip for charts
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div style={styles.chartTooltip}>
                <p style={styles.tooltipLabel}>{`Time: ${label}`}</p>
                {payload.map((entry, index) => (
                    <p key={index} style={{ color: entry.color, margin: 0, fontSize: '12px' }}>
                        {`${entry.dataKey}: ${entry.value}`}
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

// Format time for chart labels
const formatTime = (timestamp) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/* ===========================
   4) React component
   =========================== */
const WaterDetectionDashboard = () => {
    const [sensorData, setSensorData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [isConnected, setIsConnected] = useState(null);
    const [ai, setAi] = useState({ label: "-", suggestion: "-", neighbors: [] });
    const [showNN, setShowNN] = useState(false);
    
    // Historical data for charts
    const [historicalData, setHistoricalData] = useState([]);
    
    // Chart visibility toggles
    const [showCharts, setShowCharts] = useState({
        waterQuality: true,
        npk: true,
        environmental: true,
        waterLevel: false
    });

    // Initialize Firebase app once
    const app = useMemo(() => {
        try {
            return getApp(APP_NAME);
        } catch {
            return initializeApp(firebaseConfig, APP_NAME);
        }
    }, []);
    const db = useMemo(() => getDatabase(app), [app]);

    useEffect(() => {
        setLoading(true);

        // RTDB connection
        const infoRef = ref(db, ".info/connected");
        const infoUnsub = onValue(
            infoRef,
            (snap) => setIsConnected(!!snap.val()),
            () => setIsConnected(false)
        );

        // Data listener (adjust path if needed)
        const dataRef = ref(db, "/NPK");
        const dataUnsub = onValue(
            dataRef,
            (snapshot) => {
                if (snapshot.exists()) {
                    const data = snapshot.val() || {};
                    setSensorData(data);
                    const currentTime = new Date().toLocaleString();
                    setLastUpdated(currentTime);
                    setError(null);

                    // Add to historical data
                    const timestamp = Date.now();
                    const newDataPoint = {
                        timestamp,
                        time: formatTime(timestamp),
                        ph: Number(data.ph) || 0,
                        tds_ppm: Number(data.tds_ppm) || 0,
                        temp: Number(data.temp) || 0,
                        ec: Number(data.ec) || 0,
                        turbidity_ntu: Number(data.turbidity_ntu) || 0,
                        water_pct: Number(data.water_pct) || 0,
                        n: Number(data.n) || 0,
                        p: Number(data.p) || 0,
                        k: Number(data.k) || 0,
                        hum: Number(data.hum) || 0
                    };

                    setHistoricalData(prev => {
                        const updated = [...prev, newDataPoint];
                        // Keep only last MAX_DATA_POINTS
                        return updated.slice(-MAX_DATA_POINTS);
                    });

                    // Compute KNN suggestion
                    const sample = {
                        ph: data.ph,
                        tds_ppm: data.tds_ppm,
                        turbidity_ntu: data.turbidity_ntu,
                        temp: data.temp,
                        ec: data.ec,
                        water_pct: data.water_pct,
                        n: data.n,
                        p: data.p,
                        k: data.k,
                    };
                    const { neighbors, majorityLabel, suggestionText } = knnSuggest(sample, 3);
                    setAi({
                        label: majorityLabel || "-",
                        suggestion: suggestionText,
                        neighbors,
                    });
                } else {
                    setSensorData(null);
                    setError("No data available at /NPK");
                }
                setLoading(false);
            },
            (err) => {
                console.error("[RTDB] onValue error:", err);
                setError("Failed to fetch sensor data");
                setLoading(false);
            }
        );

        return () => {
            off(infoRef);
            off(dataRef);
            infoUnsub();
            dataUnsub();
        };
    }, [db]);

    const getStatusColor = (value, type) => {
        switch (type) {
            case "ph":
                if (value < 6.5) return "#b71c1c";
                if (value > 8.5) return "#b71c1c";
                return "#2e7d32";
            case "temp":
                if (value < 0 || value > 35) return "#b71c1c";
                return "#2e7d32";
            case "turbidity":
                if (value > 4) return "#b71c1c";
                return "#2e7d32";
            default:
                return "#1e88e5";
        }
    };

    const formatTimestamp = (timestamp) => {
        if (!timestamp) return "-";
        const n = Number(timestamp);
        const ms = n < 10_000_000_000 ? n * 1000 : n;
        return new Date(ms).toLocaleString();
    };

    const toggleChart = (chartType) => {
        setShowCharts(prev => ({
            ...prev,
            [chartType]: !prev[chartType]
        }));
    };

    if (loading) {
        return (
            <div style={styles.container}>
                <div style={styles.loading}>
                    <div style={styles.spinner}></div>
                    <p>Loading sensor data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div style={styles.container}>
                <header style={styles.header}>
                    <h1 style={styles.title}>Water Detection Sensor Dashboard</h1>
                </header>
                <div style={styles.error}>
                    <h2>Error</h2>
                    <p>{error}</p>
                    <button style={styles.retryButton} onClick={() => window.location.reload()}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    const s = sensorData || {};
    const {
        n = 0,
        p = 0,
        k = 0,
        ph = 0,
        tds_ppm = 0,
        tds_v = 0,
        temp = 0,
        hum = 0,
        ec = 0,
        turbidity_ntu = 0,
        turbidity_v = 0,
        water_pct = 0,
        water_v = 0,
        relay_on = false,
        ts,
    } = s;

    return (
        <div style={styles.container}>
            <header style={styles.header}>
                <h1 style={styles.title}>Water Detection Sensor Dashboard</h1>
                <div style={styles.lastUpdated}>Last Updated: {lastUpdated || "-"}</div>
                <div style={{ marginTop: 8, fontSize: 12, color: isConnected ? "#2e7d32" : "#b71c1c" }}>
                    RTDB Connection: {isConnected === null ? "…" : isConnected ? "Online" : "Offline"}
                </div>
            </header>

            <div style={styles.grid}>
                {/* NPK Values */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>NPK Levels</h3>
                    <div style={styles.npkGrid}>
                        <div style={styles.npkItem}>
                            <span style={styles.npkLabel}>N (Nitrogen)</span>
                            <span style={styles.npkValue}>{n}</span>
                        </div>
                        <div style={styles.npkItem}>
                            <span style={styles.npkLabel}>P (Phosphorus)</span>
                            <span style={styles.npkValue}>{p}</span>
                        </div>
                        <div style={styles.npkItem}>
                            <span style={styles.npkLabel}>K (Potassium)</span>
                            <span style={styles.npkValue}>{k}</span>
                        </div>
                    </div>
                </div>

                {/* Water Quality */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>Water Quality</h3>
                    <div style={styles.parameterGrid}>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>pH Level</span>
                            <span style={{ ...styles.paramValue, color: getStatusColor(ph, "ph") }}>{ph}</span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>TDS (ppm)</span>
                            <span style={styles.paramValue}>{tds_ppm}</span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>TDS Voltage</span>
                            <span style={styles.paramValue}>{tds_v}V</span>
                        </div>
                    </div>
                </div>

                {/* Environmental */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>Environmental</h3>
                    <div style={styles.parameterGrid}>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Temperature</span>
                            <span style={{ ...styles.paramValue, color: getStatusColor(temp, "temp") }}>
                                {temp}°C
                            </span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Humidity</span>
                            <span style={styles.paramValue}>{hum}%</span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>EC</span>
                            <span style={styles.paramValue}>{ec}</span>
                        </div>
                    </div>
                </div>

                {/* Turbidity */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>Turbidity</h3>
                    <div style={styles.parameterGrid}>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Turbidity (NTU)</span>
                            <span style={{ ...styles.paramValue, color: getStatusColor(turbidity_ntu, "turbidity") }}>
                                {turbidity_ntu}
                            </span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Turbidity Voltage</span>
                            <span style={styles.paramValue}>{turbidity_v}V</span>
                        </div>
                    </div>
                </div>

                {/* Water Level */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>Water Level</h3>
                    <div style={styles.parameterGrid}>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Water Percentage</span>
                            <span style={styles.paramValue}>{water_pct}%</span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Water Voltage</span>
                            <span style={styles.paramValue}>{water_v}V</span>
                        </div>
                    </div>
                </div>

                {/* System Status */}
                <div style={styles.card}>
                    <h3 style={styles.cardTitle}>System Status</h3>
                    <div style={styles.parameterGrid}>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Relay Status</span>
                            <span
                                style={{
                                    ...styles.statusBadge,
                                    backgroundColor: relay_on ? "#2e7d32" : "#b71c1c",
                                }}
                            >
                                {relay_on ? "ON" : "OFF"}
                            </span>
                        </div>
                        <div style={styles.parameter}>
                            <span style={styles.paramLabel}>Timestamp</span>
                            <span style={styles.paramValue}>{formatTimestamp(ts)}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Section */}
            <div style={styles.chartsSection}>
                <div style={styles.chartsSectionHeader}>
                    <h2 style={styles.chartsSectionTitle}>Real-time Sensor Trends</h2>
                    <div style={styles.chartToggles}>
                        <button 
                            onClick={() => toggleChart('waterQuality')} 
                            style={{...styles.toggleBtn, backgroundColor: showCharts.waterQuality ? '#1e88e5' : '#f5f5f5'}}
                        >
                            Water Quality
                        </button>
                        <button 
                            onClick={() => toggleChart('npk')} 
                            style={{...styles.toggleBtn, backgroundColor: showCharts.npk ? '#1e88e5' : '#f5f5f5'}}
                        >
                            NPK Levels
                        </button>
                        <button 
                            onClick={() => toggleChart('environmental')} 
                            style={{...styles.toggleBtn, backgroundColor: showCharts.environmental ? '#1e88e5' : '#f5f5f5'}}
                        >
                            Environmental
                        </button>
                        <button 
                            onClick={() => toggleChart('waterLevel')} 
                            style={{...styles.toggleBtn, backgroundColor: showCharts.waterLevel ? '#1e88e5' : '#f5f5f5'}}
                        >
                            Water Level
                        </button>
                    </div>
                </div>

                {/* Water Quality Chart */}
                {showCharts.waterQuality && (
                    <div style={styles.chartCard}>
                        <h3 style={styles.chartTitle}>Water Quality Trends</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={historicalData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis yAxisId="ph" orientation="left" domain={[5, 9]} />
                                <YAxis yAxisId="tds" orientation="right" domain={[0, 'dataMax']} />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                                <ReferenceLine yAxisId="ph" y={6.5} stroke="#ff9800" strokeDasharray="5 5" />
                                <ReferenceLine yAxisId="ph" y={7.5} stroke="#ff9800" strokeDasharray="5 5" />
                                <Line yAxisId="ph" type="monotone" dataKey="ph" stroke="#e53e3e" strokeWidth={2} name="pH" />
                                <Line yAxisId="tds" type="monotone" dataKey="tds_ppm" stroke="#3182ce" strokeWidth={2} name="TDS (ppm)" />
                                <Line yAxisId="tds" type="monotone" dataKey="ec" stroke="#805ad5" strokeWidth={2} name="EC" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* NPK Chart */}
                {showCharts.npk && (
                    <div style={styles.chartCard}>
                        <h3 style={styles.chartTitle}>NPK Nutrient Levels</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={historicalData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                                <Area type="monotone" dataKey="n" stackId="1" stroke="#38a169" fill="#38a169" fillOpacity={0.6} name="Nitrogen (N)" />
                                <Area type="monotone" dataKey="p" stackId="1" stroke="#d69e2e" fill="#d69e2e" fillOpacity={0.6} name="Phosphorus (P)" />
                                <Area type="monotone" dataKey="k" stackId="1" stroke="#e53e3e" fill="#e53e3e" fillOpacity={0.6} name="Potassium (K)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Environmental Chart */}
                {showCharts.environmental && (
                    <div style={styles.chartCard}>
                        <h3 style={styles.chartTitle}>Environmental Conditions</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={historicalData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis yAxisId="temp" orientation="left" domain={[15, 35]} />
                                <YAxis yAxisId="hum" orientation="right" domain={[0, 100]} />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                                <ReferenceLine yAxisId="temp" y={25} stroke="#ff9800" strokeDasharray="5 5" />
                                <Line yAxisId="temp" type="monotone" dataKey="temp" stroke="#e53e3e" strokeWidth={2} name="Temperature (°C)" />
                                <Line yAxisId="hum" type="monotone" dataKey="hum" stroke="#3182ce" strokeWidth={2} name="Humidity (%)" />
                                <Line yAxisId="temp" type="monotone" dataKey="turbidity_ntu" stroke="#805ad5" strokeWidth={2} name="Turbidity (NTU)" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Water Level Chart */}
                {showCharts.waterLevel && (
                    <div style={styles.chartCard}>
                        <h3 style={styles.chartTitle}>Water Level Monitoring</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={historicalData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" />
                                <YAxis domain={[0, 100]} />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                                <ReferenceLine y={20} stroke="#e53e3e" strokeDasharray="5 5" />
                                <ReferenceLine y={80} stroke="#38a169" strokeDasharray="5 5" />
                                <Area type="monotone" dataKey="water_pct" stroke="#3182ce" fill="#3182ce" fillOpacity={0.6} name="Water Level (%)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>

            {/* AI Suggestions (KNN) - Full Width */}
            <div style={styles.fullWidthCard}>
                <div style={styles.cardHeaderRow}>
                    <h3 style={styles.cardTitle}>AI Suggestions (KNN)</h3>
                    <span style={styles.actionPill}>
                        {ai.label || "-"}
                    </span>
                </div>

                <div style={{ display: "grid", gap: 14 }}>
                    <div style={styles.kvRow}>
                        <span style={styles.kvLabel}>Recommended Action</span>
                        <span style={styles.kvValue}>{ai.label || "-"}</span>
                    </div>

                    <div style={styles.suggestionBox}>
                        <div style={styles.suggestionTitle}>Suggestion</div>
                        <div style={styles.suggestionText}>{ai.suggestion}</div>
                    </div>

                    <div style={styles.metaChips}>
                        <span style={styles.outlineChip}>Model: k-NN (k=3)</span>
                        <span style={styles.outlineChip}>
                            Neighbors: {Math.max(0, ai.neighbors?.length || 0)}
                        </span>
                        <span style={styles.outlineChip}>
                            Data Points: {historicalData.length}
                        </span>
                    </div>

                    <button
                        onClick={() => setShowNN((v) => !v)}
                        style={styles.outlinedBtn}
                    >
                        {showNN ? "Hide Nearest Neighbors" : "Show Nearest Neighbors"}
                    </button>

                    {showNN && (
                        <div style={styles.nnList}>
                            {(ai.neighbors || []).map((n, i) => (
                                <div key={i} style={styles.nnItem}>
                                    <div style={styles.nnItemHeader}>
                                        <span>Neighbor {i + 1}</span>
                                        <span style={styles.nnLabelBadge}>{n.row.label}</span>
                                    </div>
                                    <div style={styles.nnBodyGrid}>
                                        <div>
                                            <div style={styles.nnMetric}>
                                                <span>Distance²</span>
                                                <strong>{n.d2.toFixed(4)}</strong>
                                            </div>
                                            <div style={styles.nnMetric}>
                                                <span>Reason</span>
                                                <strong>{n.row.suggestion}</strong>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

/* ===========================
   5) Styles
   =========================== */
const styles = {
    container: {
        minHeight: "100vh",
        backgroundColor: "#f5f5f5",
        padding: "20px",
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    header: {
        marginBottom: "20px",
        textAlign: "center",
    },
    title: {
        color: "#333",
        marginBottom: "10px",
        fontSize: "2.2rem",
        fontWeight: "500",
    },
    lastUpdated: {
        color: "#666",
        fontSize: "14px",
    },
    grid: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
        gap: "20px",
        maxWidth: "1200px",
        margin: "0 auto 20px auto",
    },
    card: {
        backgroundColor: "white",
        borderRadius: "12px",
        padding: "24px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
        border: "1px solid #e0e0e0",
    },
    fullWidthCard: {
        backgroundColor: "white",
        borderRadius: "12px",
        padding: "24px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
        border: "1px solid #e0e0e0",
        maxWidth: "1200px",
        margin: "0 auto",
        width: "100%",
    },
    chartsSection: {
        maxWidth: "1200px",
        margin: "0 auto 20px auto",
        width: "100%",
    },
    chartsSectionHeader: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "20px",
        flexWrap: "wrap",
        gap: "10px",
    },
    chartsSectionTitle: {
        color: "#333",
        fontSize: "1.6rem",
        fontWeight: "600",
        margin: 0,
    },
    chartToggles: {
        display: "flex",
        gap: "8px",
        flexWrap: "wrap",
    },
    toggleBtn: {
        padding: "8px 16px",
        border: "1px solid #e0e0e0",
        borderRadius: "6px",
        cursor: "pointer",
        fontSize: "14px",
        fontWeight: "500",
        transition: "all 0.2s",
        color: "#333",
    },
    chartCard: {
        backgroundColor: "white",
        borderRadius: "12px",
        padding: "24px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
        border: "1px solid #e0e0e0",
        marginBottom: "20px",
    },
    chartTitle: {
        margin: "0 0 16px 0",
        color: "#333",
        fontSize: "1.1rem",
        fontWeight: "600",
        borderBottom: "2px solid #1e88e5",
        paddingBottom: "6px",
    },
    chartTooltip: {
        backgroundColor: "white",
        border: "1px solid #e0e0e0",
        borderRadius: "6px",
        padding: "8px 12px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    },
    tooltipLabel: {
        fontSize: "12px",
        fontWeight: "600",
        margin: "0 0 4px 0",
        color: "#333",
    },
    cardHeaderRow: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 12,
    },
    actionPill: {
        backgroundColor: "#e8f0fe",
        color: "#1e88e5",
        border: "1px solid #bbdefb",
        borderRadius: 20,
        padding: "6px 12px",
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: 0.2,
        textTransform: "capitalize",
    },
    kvRow: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 14px",
        backgroundColor: "#f8f9fa",
        borderRadius: 8,
        border: "1px solid #dee2e6",
    },
    kvLabel: {
        color: "#555",
        fontSize: 14,
        fontWeight: 600,
    },
    kvValue: {
        color: "#222",
        fontSize: 15,
        fontWeight: 700,
    },
    suggestionBox: {
        border: "1px solid #e0e0e0",
        background: "#fff",
        borderRadius: 10,
        padding: 14,
    },
    suggestionTitle: {
        fontSize: 12,
        fontWeight: 700,
        color: "#6b7280",
        marginBottom: 6,
        textTransform: "uppercase",
        letterSpacing: 0.6,
    },
    suggestionText: {
        fontSize: 14,
        lineHeight: 1.5,
        color: "#222",
        background: "#f8fafc",
        border: "1px dashed #e2e8f0",
        padding: 12,
        borderRadius: 8,
    },
    outlinedBtn: {
        alignSelf: "flex-start",
        background: "#fff",
        color: "#1e88e5",
        border: "1px solid #1e88e5",
        borderRadius: 8,
        padding: "10px 14px",
        cursor: "pointer",
        fontSize: 14,
        fontWeight: 600,
    },
    metaChips: {
        display: "flex",
        gap: 8,
        flexWrap: "wrap",
    },
    outlineChip: {
        border: "1px solid #e0e0e0",
        color: "#555",
        background: "#fff",
        borderRadius: 999,
        padding: "6px 10px",
        fontSize: 12,
        fontWeight: 600,
    },
    nnList: {
        display: "grid",
        gap: 10,
    },
    nnItem: {
        border: "1px solid #e5e7eb",
        background: "#f9fafb",
        borderRadius: 10,
        padding: 12,
    },
    nnItemHeader: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 8,
        color: "#374151",
        fontSize: 13,
        fontWeight: 700,
    },
    nnLabelBadge: {
        backgroundColor: "#e8f0fe",
        color: "#1e88e5",
        border: "1px solid #bbdefb",
        borderRadius: 999,
        padding: "4px 10px",
        fontSize: 11,
        fontWeight: 700,
    },
    nnBodyGrid: {
        display: "grid",
        gap: 8,
    },
    nnMetric: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontSize: 13,
        color: "#4b5563",
        background: "#fff",
        padding: "8px 10px",
        borderRadius: 8,
        border: "1px solid #e5e7eb",
    },
    cardTitle: {
        margin: "0 0 16px 0",
        color: "#333",
        fontSize: "1.1rem",
        fontWeight: "600",
        borderBottom: "2px solid #1e88e5",
        paddingBottom: "6px",
    },
    npkGrid: {
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "16px",
    },
    npkItem: {
        textAlign: "center",
        padding: "16px",
        backgroundColor: "#f8f9fa",
        borderRadius: "8px",
        border: "1px solid #dee2e6",
    },
    npkLabel: {
        display: "block",
        fontSize: "12px",
        color: "#666",
        marginBottom: "8px",
        fontWeight: "500",
    },
    npkValue: {
        display: "block",
        fontSize: "22px",
        fontWeight: "700",
        color: "#1e88e5",
    },
    parameterGrid: {
        display: "flex",
        flexDirection: "column",
        gap: "12px",
    },
    parameter: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "12px 14px",
        backgroundColor: "#f8f9fa",
        borderRadius: "8px",
        border: "1px solid #dee2e6",
    },
    paramLabel: {
        color: "#555",
        fontSize: "14px",
        fontWeight: "600",
    },
    paramValue: {
        color: "#222",
        fontSize: "15px",
        fontWeight: "700",
    },
    statusBadge: {
        padding: "4px 12px",
        borderRadius: "20px",
        color: "white",
        fontSize: "12px",
        fontWeight: "bold",
        textAlign: "center",
    },
    loading: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "50vh",
        color: "#666",
    },
    spinner: {
        width: "40px",
        height: "40px",
        border: "4px solid #f3f3f3",
        borderTop: "4px solid #1e88e5",
        borderRadius: "50%",
        animation: "spin 1s linear infinite",
        marginBottom: "16px",
    },
    error: {
        textAlign: "center",
        padding: "40px",
        color: "#b71c1c",
    },
    retryButton: {
        backgroundColor: "#1e88e5",
        color: "white",
        border: "none",
        padding: "12px 24px",
        borderRadius: "6px",
        cursor: "pointer",
        fontSize: "16px",
        marginTop: "16px",
    },
};

// Inject keyframes for spinner animation
const styleEl = document.createElement("style");
styleEl.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(styleEl);

export default WaterDetectionDashboard;