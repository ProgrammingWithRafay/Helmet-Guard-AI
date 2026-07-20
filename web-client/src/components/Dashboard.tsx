"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  AlertTriangle,
  Camera,
  CameraOff,
  CheckCircle,
  ChevronRight,
  HardHat,
  Radio,
  ShieldAlert,
  Wifi,
  WifiOff,
  XCircle,
  Activity,
  Clock,
  Upload
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const API_BASE = "http://localhost:8000/api/v1";

interface BBox {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
}

interface Detection {
    id: string;
    class: string;
    confidence: number;
    bbox: BBox;
}

interface Summary {
    total_riders: number;
    compliant: number;
    violations: number;
}

interface AlertEntry {
  id: number;
  ts: string;
  message: string;
  severity: "critical" | "warning" | "info";
  count: number;
}

interface ChartPoint {
  time: string;
  compliance: number;
  total: number;
  violations: number;
}

function fmtTime(iso: string) {
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function complianceColor(pct: number) {
  if (pct >= 90) return "#00d9a0";
  if (pct >= 70) return "#f59e0b";
  return "#ff4444";
}

function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  color = "#00d9a0",
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  color?: string;
}) {
  return (
    <div className="bg-card border border-border rounded-sm p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground font-mono text-xs tracking-widest uppercase">
          {label}
        </span>
        <Icon size={14} style={{ color }} />
      </div>
      <div className="text-3xl font-semibold tracking-tight" style={{ color }}>
        {value}
      </div>
      {sub && (
        <div className="text-muted-foreground font-mono text-xs">{sub}</div>
      )}
    </div>
  );
}

function ConnectionBadge({ connected }: { connected: boolean }) {
  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1 rounded-sm font-mono text-xs border ${
        connected
          ? "border-[#00d9a030] bg-[#00d9a010] text-[#00d9a0]"
          : "border-[#ff444430] bg-[#ff444410] text-[#ff4444]"
      }`}
    >
      {connected ? <Wifi size={11} /> : <WifiOff size={11} />}
      {connected ? "HTTP API UP" : "OFFLINE"}
    </div>
  );
}

function AlertRow({ alert }: { alert: AlertEntry }) {
  const colors = {
    critical: { text: "#ff4444", bg: "#ff444415", border: "#ff444430" },
    warning: { text: "#f59e0b", bg: "#f59e0b15", border: "#f59e0b30" },
    info: { text: "#6b7f8e", bg: "#6b7f8e10", border: "#6b7f8e20" },
  };
  const c = colors[alert.severity];
  const Icon = alert.severity === "critical" ? ShieldAlert : alert.severity === "warning" ? AlertTriangle : CheckCircle;

  return (
    <div
      className="flex items-start gap-3 px-3 py-2.5 border-b border-border/60 hover:bg-secondary/40 transition-colors"
      style={{ borderLeft: `2px solid ${c.border}` }}
    >
      <Icon size={13} className="mt-0.5 shrink-0" style={{ color: c.text }} />
      <div className="flex-1 min-w-0">
        <div className="font-mono text-xs" style={{ color: c.text }}>
          {alert.message}
        </div>
        <div className="text-muted-foreground font-mono text-[10px] mt-0.5">{alert.ts}</div>
      </div>
      {alert.count > 1 && (
        <div
          className="shrink-0 font-mono text-[10px] px-1.5 py-0.5 rounded-sm"
          style={{ background: c.bg, color: c.text }}
        >
          ×{alert.count}
        </div>
      )}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-card border border-border px-3 py-2 rounded-sm font-mono text-xs">
      <div className="text-muted-foreground mb-1">{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value}
          {p.name === "compliance" ? "%" : ""}
        </div>
      ))}
    </div>
  );
};


export default function Dashboard() {
    const [mode, setMode] = useState<'upload' | 'webcam'>('webcam');
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [summary, setSummary] = useState<Summary>({ total_riders: 0, compliant: 0, violations: 0 });
    const [confidence, setConfidenceState] = useState<number>(0.45);
    const confidenceRef = useRef<number>(0.45);
    
    const setConfidence = (val: number) => {
        setConfidenceState(val);
        confidenceRef.current = val;
    };
    
    const [isDetecting, setIsDetecting] = useState<boolean>(false);
    
    const [cameraOn, setCameraOn] = useState(false);
    const [alerts, setAlerts] = useState<AlertEntry[]>([]);
    const [chartData, setChartData] = useState<ChartPoint[]>([]);
    const [latency, setLatency] = useState<number>(0);
    const [frameCount, setFrameCount] = useState(0);
    const [sessionStart] = useState(() => Date.now());
    const [uptime, setUptime] = useState("00:00:00");

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const isDetectingRef = useRef<boolean>(false);
    const alertIdRef = useRef(0);

    // Stop webcam when unmounting or switching modes
    useEffect(() => {
        if (mode === 'upload') {
            stopCamera();
            setDetections([]);
            setSummary({ total_riders: 0, compliant: 0, violations: 0 });
            setImageSrc(null);
            clearCanvas();
        }
        return () => stopCamera();
    }, [mode]);

    // Automatically re-detect if confidence slider changes in upload mode
    useEffect(() => {
        if (mode === 'upload' && imageSrc) {
            const reDetect = async () => {
                try {
                    const res = await axios.post(`${API_BASE}/detect/frame`, {
                        frame_base64: imageSrc,
                        confidence_threshold: confidence
                    });
                    setDetections(res.data.detections);
                    setSummary(res.data.summary);
                } catch (err) {
                    console.error("Redetection failed", err);
                }
            };
            reDetect();
        }
    }, [confidence, mode, imageSrc]);

    // Draw bounding boxes when detections update
    useEffect(() => {
        drawDetections();
    }, [detections, imageSrc]);

    // Session uptime
    useEffect(() => {
        if (!isDetecting) return;
        const t = setInterval(() => {
            const s = Math.floor((Date.now() - sessionStart) / 1000);
            const h = String(Math.floor(s / 3600)).padStart(2, "0");
            const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
            const sec = String(s % 60).padStart(2, "0");
            setUptime(`${h}:${m}:${sec}`);
        }, 1000);
        return () => clearInterval(t);
    }, [isDetecting, sessionStart]);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
                setCameraOn(true);
                setMode('webcam');
            }
        } catch (err: any) {
            console.error("Error accessing webcam", err);
            if (err.name === 'NotAllowedError') {
                pushAlert("Camera permission denied. Please allow access in your browser.", "warning");
            } else {
                pushAlert(`Camera error: ${err.message || "Unknown error"}`, "critical");
            }
        }
    };

    const stopCamera = () => {
        isDetectingRef.current = false;
        setIsDetecting(false);
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        setCameraOn(false);
        setDetections([]);
        setSummary({ total_riders: 0, compliant: 0, violations: 0 });
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx?.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    const pushAlert = (message: string, severity: AlertEntry["severity"]) => {
        setAlerts((prev) => {
            if (prev.length > 0 && prev[0].message === message) {
                return [{ ...prev[0], count: prev[0].count + 1 }, ...prev.slice(1)];
            }
            const entry: AlertEntry = {
                id: ++alertIdRef.current,
                ts: new Date().toLocaleTimeString("en-US", { hour12: false }),
                message,
                severity,
                count: 1,
            };
            return [entry, ...prev].slice(0, 80);
        });
    };

    const captureAndDetect = async () => {
        if (!videoRef.current || !isDetectingRef.current) return;
        
        // Use a temporary offscreen canvas just to grab the image bytes
        const offscreen = document.createElement('canvas');
        offscreen.width = videoRef.current.videoWidth || 640;
        offscreen.height = videoRef.current.videoHeight || 480;
        const ctx = offscreen.getContext('2d');
        if (!ctx) return;
        
        ctx.drawImage(videoRef.current, 0, 0, offscreen.width, offscreen.height);
        const base64Frame = offscreen.toDataURL('image/jpeg', 0.8);

        const t0 = performance.now();
        try {
            const res = await axios.post(`${API_BASE}/detect/frame`, {
                frame_base64: base64Frame,
                confidence_threshold: confidenceRef.current
            });
            if (isDetectingRef.current) {
                setDetections(res.data.detections);
                setSummary(res.data.summary);
                setLatency(Math.round(performance.now() - t0));
                setFrameCount(n => n + 1);

                const total = res.data.summary.total_riders;
                const violations = res.data.summary.violations;
                const with_helmet = res.data.summary.compliant;
                const compPct = total > 0 ? (with_helmet / total) * 100 : 100;

                const point: ChartPoint = {
                    time: fmtTime(new Date().toISOString()),
                    compliance: Number(compPct.toFixed(1)),
                    total: total,
                    violations: violations,
                };
                setChartData((prev) => {
                    const next = [...prev, point];
                    return next.slice(-40);
                });

                if (violations > 0) {
                    pushAlert(`${violations} worker${violations > 1 ? "s" : ""} detected without helmet`, "critical");
                } else if (total > 0 && with_helmet === total) {
                    pushAlert("All workers compliant — helmets confirmed", "info");
                }
            }
        } catch (err) {
            console.error("Webcam detection failed", err);
        }
    };

    const runInferenceLoop = async () => {
        if (!isDetectingRef.current) return;
        await captureAndDetect();
        if (isDetectingRef.current) {
            requestAnimationFrame(runInferenceLoop);
        }
    };

    const toggleStream = () => {
        if (isDetecting) {
            isDetectingRef.current = false;
            setIsDetecting(false);
        } else {
            if (!cameraOn) startCamera();
            isDetectingRef.current = true;
            setIsDetecting(true);
            pushAlert("Detection session started", "info");
            runInferenceLoop();
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;
        const file = e.target.files[0];
        
        stopCamera();
        setMode('upload');
        
        const reader = new FileReader();
        reader.onload = (event) => {
            setImageSrc(event.target?.result as string);
        };
        reader.readAsDataURL(file);

        setIsDetecting(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("confidence_threshold", confidence.toString());

        const t0 = performance.now();
        try {
            const res = await axios.post(`${API_BASE}/detect/image`, formData);
            setDetections(res.data.detections);
            setSummary(res.data.summary);
            setLatency(Math.round(performance.now() - t0));
        } catch (err) {
            console.error("Detection failed", err);
        } finally {
            setIsDetecting(false);
        }
    };

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx?.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    const drawDetections = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!canvas || !ctx) return;

        const drawBoxes = (w: number, h: number, sourceElem?: HTMLImageElement | HTMLVideoElement) => {
            canvas.width = w;
            canvas.height = h;
            ctx.clearRect(0, 0, w, h);
            
            if (sourceElem) {
                ctx.drawImage(sourceElem, 0, 0, w, h);
            }

            detections.forEach(det => {
                const { x_min, y_min, x_max, y_max } = det.bbox;
                const isCompliant = det.class === 'helmet';
                let color = isCompliant ? '#00d9a0' : '#ff4444';

                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(x_min, y_min, x_max - x_min, y_max - y_min);

                ctx.fillStyle = color;
                const label = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(x_min, y_min - 24, textWidth + 10, 24);

                ctx.fillStyle = '#000000';
                ctx.font = '12px monospace';
                ctx.fillText(label, x_min + 5, y_min - 7);
            });
        };

        if (mode === 'upload' && imageSrc) {
            const img = new Image();
            img.onload = () => drawBoxes(img.width, img.height, img);
            img.src = imageSrc;
        } else if (mode === 'webcam' && videoRef.current && videoRef.current.videoWidth) {
            drawBoxes(videoRef.current.videoWidth, videoRef.current.videoHeight);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    const compPct = summary.total_riders > 0 ? (summary.compliant / summary.total_riders) * 100 : 100;
    const rollingComp = chartData.length > 0 
        ? chartData.reduce((acc, curr) => acc + curr.compliance, 0) / chartData.length 
        : compPct;
    const compColor = complianceColor(rollingComp);

    const distData = summary.total_riders > 0
      ? [
          { name: "Helmet On", value: summary.compliant, color: "#00d9a0" },
          { name: "No Helmet", value: summary.violations, color: "#ff4444" },
        ]
      : [];

    return (
        <div
          className="min-h-screen bg-background text-foreground"
          style={{ fontFamily: "'Barlow', sans-serif" }}
        >
          {/* Header */}
          <header className="border-b border-border px-6 py-3 flex items-center justify-between bg-card">
            <div className="flex items-center gap-3">
              <HardHat size={20} className="text-[#00d9a0]" />
              <div>
                <h1
                  className="text-base font-semibold tracking-wide text-foreground"
                  style={{ letterSpacing: "0.08em" }}
                >
                  HELMET DETECTION SYSTEM
                </h1>
                <div className="text-muted-foreground font-mono text-[10px] tracking-widest">
                  SAFETY COMPLIANCE MONITOR · v1.0
                </div>
              </div>
            </div>
    
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
                <Clock size={11} />
                {uptime}
              </div>
              <div className="flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
                <Activity size={11} />
                {frameCount.toLocaleString()} frames
              </div>
              <ConnectionBadge connected={true} />
    
              {/* Confidence Slider */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-sm border border-border">
                  <span className="font-mono text-[10px] text-muted-foreground w-[60px]">CONF: {Math.round(confidence * 100)}%</span>
                  <input 
                      type="range" 
                      min="10" 
                      max="100" 
                      value={Math.round(confidence * 100)} 
                      onChange={(e) => setConfidence(parseInt(e.target.value) / 100)}
                      className="w-16 h-1 bg-border rounded-lg appearance-none cursor-pointer accent-[#00d9a0]"
                  />
              </div>

              {/* Image Upload */}
              <label className="flex items-center gap-1.5 px-3 py-1.5 rounded-sm font-mono text-xs border border-border text-muted-foreground hover:border-[#00d9a040] hover:text-[#00d9a0] cursor-pointer transition-all">
                  <Upload size={12} />
                  UPLOAD IMAGE
                  <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
              </label>

              {/* Camera toggle */}
              <button
                onClick={cameraOn ? stopCamera : startCamera}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm font-mono text-xs border transition-all ${
                  cameraOn
                    ? "border-[#00d9a030] bg-[#00d9a015] text-[#00d9a0] hover:bg-[#00d9a025]"
                    : "border-border text-muted-foreground hover:border-[#00d9a040] hover:text-[#00d9a0]"
                }`}
              >
                {cameraOn ? <Camera size={12} /> : <CameraOff size={12} />}
                {cameraOn ? "CAMERA ON" : "START CAMERA"}
              </button>
    
              {/* Stream toggle */}
              <button
                onClick={toggleStream}
                disabled={mode === 'upload'}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm font-mono text-xs border transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                  isDetecting
                    ? "border-[#ff444430] bg-[#ff444415] text-[#ff4444] hover:bg-[#ff444425]"
                    : "border-[#00d9a030] bg-[#00d9a015] text-[#00d9a0] hover:bg-[#00d9a025]"
                }`}
              >
                <Radio size={12} className={isDetecting ? "animate-pulse" : ""} />
                {isDetecting ? "STOP DETECTION" : "START DETECTION"}
              </button>
            </div>
          </header>
    
          {/* Main grid */}
          <div className="grid grid-cols-[1fr_340px] gap-0 h-[calc(100vh-57px)]">
    
            {/* Left column */}
            <div className="flex flex-col overflow-hidden border-r border-border">
    
              {/* Video area */}
              <div className="relative bg-[#050709] flex-1 flex items-center justify-center overflow-hidden">
                <video ref={videoRef} className={mode === 'webcam' && cameraOn ? "absolute inset-0 w-full h-full object-contain" : "hidden"} playsInline muted autoPlay />
                
                {/* Result display */}
                {mode === 'webcam' || mode === 'upload' ? (
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full object-contain z-10 pointer-events-none"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-4 text-muted-foreground">
                    <CameraOff size={40} strokeWidth={1} />
                    <div className="font-mono text-sm text-center">
                      <div>Camera not active</div>
                      <div className="text-xs mt-1 opacity-60">Click "START CAMERA" or "UPLOAD IMAGE" to begin</div>
                    </div>
                  </div>
                )}
    
                {/* Overlay badges */}
                {isDetecting && (
                  <div className="absolute top-3 left-3 flex flex-col gap-1.5">
                    <div className="flex items-center gap-1.5 bg-black/70 backdrop-blur-sm px-2 py-1 rounded-sm font-mono text-[10px] text-[#00d9a0] border border-[#00d9a030]">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#00d9a0] animate-pulse inline-block" />
                      LIVE DETECTION
                    </div>
                    <div className="bg-black/70 backdrop-blur-sm px-2 py-1 rounded-sm font-mono text-[10px] text-muted-foreground border border-border">
                      {latency}ms latency
                    </div>
                  </div>
                )}
    
                {/* Compliance badge */}
                {(isDetecting || mode === 'upload') && summary.total_riders > 0 && (
                  <div className="absolute top-3 right-3 bg-black/80 backdrop-blur-sm border border-border rounded-sm p-3 min-w-[100px]">
                    <div className="font-mono text-[9px] text-muted-foreground tracking-widest mb-1">
                      COMPLIANCE
                    </div>
                    <div
                      className="text-3xl font-semibold"
                      style={{ color: compColor }}
                    >
                      {compPct.toFixed(0)}%
                    </div>
                  </div>
                )}
    
                {/* Per-detection labels overlay */}
                {(isDetecting || mode === 'upload') && detections.length > 0 && (
                  <div className="absolute bottom-3 left-3 flex flex-col gap-1">
                    {detections.map((d, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-1.5 bg-black/75 backdrop-blur-sm px-2 py-1 rounded-sm font-mono text-[10px] border"
                        style={{
                          borderColor: d.class === 'helmet' ? "#00d9a030" : "#ff444430",
                          color: d.class === 'helmet' ? "#00d9a0" : "#ff4444",
                        }}
                      >
                        {d.class === 'helmet' ? <CheckCircle size={10} /> : <XCircle size={10} />}
                        {d.class} · {(d.confidence * 100).toFixed(0)}%
                      </div>
                    ))}
                  </div>
                )}
              </div>
    
              {/* Compliance trend chart */}
              <div className="h-[160px] border-t border-border bg-card px-4 pt-3 pb-2">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-[10px] text-muted-foreground tracking-widest">
                    COMPLIANCE TREND
                  </span>
                  <span className="font-mono text-[10px]" style={{ color: compColor }}>
                    {rollingComp.toFixed(1)}% rolling avg
                  </span>
                </div>
                <ResponsiveContainer width="100%" height={110}>
                  <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 0, left: -28 }}>
                    <defs>
                      <linearGradient id="compGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00d9a0" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#00d9a0" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="time" tick={{ fill: "#4a5a66", fontSize: 9, fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} tick={{ fill: "#4a5a66", fontSize: 9, fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="compliance" name="compliance" stroke="#00d9a0" strokeWidth={1.5} fill="url(#compGrad)" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
    
            {/* Right panel */}
            <div className="flex flex-col overflow-hidden bg-background">
    
              {/* Stat cards */}
              <div className="grid grid-cols-2 gap-px border-b border-border bg-border">
                <div className="bg-background">
                  <StatCard
                    label="Detected"
                    value={summary.total_riders}
                    sub="this frame"
                    icon={Activity}
                    color="#00d9a0"
                  />
                </div>
                <div className="bg-background">
                  <StatCard
                    label="Violations"
                    value={summary.violations}
                    sub="no helmet"
                    icon={ShieldAlert}
                    color={summary.violations > 0 ? "#ff4444" : "#6b7f8e"}
                  />
                </div>
                <div className="bg-background">
                  <StatCard
                    label="Compliant"
                    value={summary.compliant}
                    sub="with helmet"
                    icon={CheckCircle}
                    color="#00d9a0"
                  />
                </div>
                <div className="bg-background">
                  <StatCard
                    label="Latency"
                    value={latency > 0 ? `${latency}ms` : "--"}
                    sub="inference"
                    icon={Radio}
                    color="#f59e0b"
                  />
                </div>
              </div>
    
              {/* Distribution mini-chart */}
              <div className="px-4 pt-3 pb-2 border-b border-border">
                <div className="font-mono text-[10px] text-muted-foreground tracking-widest mb-2">
                  FRAME DISTRIBUTION
                </div>
                {distData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={60}>
                    <BarChart data={distData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                      <XAxis dataKey="name" tick={{ fill: "#6b7f8e", fontSize: 9, fontFamily: "JetBrains Mono" }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fill: "#6b7f8e", fontSize: 9 }} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                        {distData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-[60px] flex items-center justify-center font-mono text-[10px] text-muted-foreground/40">
                    No detections yet
                  </div>
                )}
              </div>
    
              {/* Alert log */}
              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="px-4 py-2.5 border-b border-border flex items-center justify-between">
                  <span className="font-mono text-[10px] text-muted-foreground tracking-widest">
                    ALERT LOG
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {alerts.length} events
                    </span>
                    {alerts.length > 0 && (
                      <button
                        onClick={() => setAlerts([])}
                        className="font-mono text-[9px] text-muted-foreground/60 hover:text-muted-foreground border border-border px-1.5 py-0.5 rounded-sm transition-colors"
                      >
                        CLEAR
                      </button>
                    )}
                  </div>
                </div>
    
                <div className="flex-1 overflow-y-auto scrollbar-hide">
                  {alerts.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 gap-2 text-muted-foreground/40">
                      <ShieldAlert size={24} strokeWidth={1} />
                      <span className="font-mono text-[10px]">No alerts recorded</span>
                    </div>
                  ) : (
                    alerts.map((a) => <AlertRow key={a.id} alert={a} />)
                  )}
                </div>
              </div>
    
              {/* Footer status */}
              <div className="border-t border-border px-4 py-2 flex items-center justify-between">
                <div className="font-mono text-[9px] text-muted-foreground/50 tracking-widest">
                  API: {API_BASE}
                </div>
                <ChevronRight size={10} className="text-muted-foreground/30" />
              </div>
            </div>
          </div>
    
          <style>{`
            .scrollbar-hide::-webkit-scrollbar { display: none; }
            .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
          `}</style>
        </div>
    );
}
