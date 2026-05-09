# ==========================================================
# QUANTUM AI MEGA LANDING PAGE (FLASK)
# ----------------------------------------------------------
# Features:
# ✔ Advanced animated landing page
# ✔ AI + Quantum Computing + Sensing + Communication
# ✔ Glassmorphism UI
# ✔ Responsive layout
# ✔ Strange quantum effects:
#     - superposition particles
#     - entanglement beams
#     - electromagnetic pulses
#     - sentient AI eye
# ✔ Smooth scroll sections
# ✔ Modern CSS styling
#
# RUN:
# python app.py
#
# OPEN:
# http://127.0.0.1:5000/
# ==========================================================

from flask import Flask, render_template_string
print("SY-5, Kevin Victor, Roll No.-30")
app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum AI Nexus</title>

<style>
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    scroll-behavior:smooth;
}

body{
    font-family:Arial, Helvetica, sans-serif;
    background:
        radial-gradient(circle at top left,#00e5ff22,transparent 25%),
        radial-gradient(circle at bottom right,#7c4dff22,transparent 25%),
        linear-gradient(135deg,#05060a,#0a1020,#111827);
    color:white;
    overflow-x:hidden;
}

/* =====================================================
   FLOATING PARTICLES (SUPERPOSITION)
===================================================== */
.particle{
    position:fixed;
    width:10px;
    height:10px;
    border-radius:50%;
    background:cyan;
    box-shadow:0 0 20px cyan;
    opacity:.55;
    animation:floatParticle linear infinite;
}

.p1{left:8%; animation-duration:8s;}
.p2{left:22%; animation-duration:11s;}
.p3{left:38%; animation-duration:7s;}
.p4{left:58%; animation-duration:13s;}
.p5{left:78%; animation-duration:9s;}
.p6{left:90%; animation-duration:10s;}

@keyframes floatParticle{
    from{transform:translateY(105vh) scale(.5);}
    to{transform:translateY(-15vh) scale(1.3);}
}

/* =====================================================
   HERO
===================================================== */
.hero{
    min-height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    text-align:center;
    padding:40px;
    position:relative;
}

.glass{
    max-width:950px;
    padding:40px;
    border-radius:28px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.18);
    backdrop-filter:blur(18px);
    box-shadow:0 20px 50px rgba(0,0,0,.45);
    animation:rise 1.4s ease;
}

@keyframes rise{
    from{opacity:0; transform:translateY(60px);}
    to{opacity:1; transform:translateY(0);}
}

.hero h1{
    font-size:64px;
    line-height:1.05;
    margin-bottom:18px;
    background:linear-gradient(90deg,#00e5ff,#8e7dff,#00ff95);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p{
    color:#d6e3ff;
    font-size:20px;
    line-height:1.6;
    margin-bottom:30px;
}

.btn{
    display:inline-block;
    padding:15px 28px;
    border-radius:14px;
    text-decoration:none;
    color:white;
    background:linear-gradient(135deg,#00c6ff,#0072ff);
    font-weight:bold;
    transition:.3s;
}

.btn:hover{
    transform:translateY(-3px) scale(1.02);
    box-shadow:0 18px 30px rgba(0,114,255,.35);
}

/* =====================================================
   AI SENTIENT EYE
===================================================== */
.eye-wrap{
    margin:28px auto 0;
    width:180px;
    height:180px;
    border-radius:50%;
    border:2px solid rgba(255,255,255,.15);
    display:flex;
    align-items:center;
    justify-content:center;
    position:relative;
    animation:rotateRing 12s linear infinite;
}

@keyframes rotateRing{
    from{transform:rotate(0deg);}
    to{transform:rotate(360deg);}
}

.eye{
    width:92px;
    height:92px;
    border-radius:50%;
    background:radial-gradient(circle,#00e5ff,#001b33);
    box-shadow:0 0 35px #00e5ff;
    position:relative;
}

.eye::before{
    content:"";
    position:absolute;
    width:30px;
    height:30px;
    border-radius:50%;
    background:white;
    top:32px;
    left:31px;
    animation:blink 4s infinite;
}

@keyframes blink{
    0%,45%,55%,100%{transform:scaleY(1);}
    50%{transform:scaleY(.08);}
}

/* =====================================================
   ENTANGLEMENT BEAM
===================================================== */
.beam{
    width:100%;
    height:3px;
    background:linear-gradient(90deg,transparent,#00ffff,transparent);
    margin:0 auto;
    opacity:.7;
    animation:pulseBeam 2s infinite;
}

@keyframes pulseBeam{
    0%,100%{transform:scaleX(.4);}
    50%{transform:scaleX(1);}
}

/* =====================================================
   SECTION
===================================================== */
.section{
    padding:80px 20px;
}

.grid{
    max-width:1200px;
    margin:auto;
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
    gap:24px;
}

.card{
    padding:28px;
    border-radius:22px;
    background:rgba(255,255,255,.07);
    border:1px solid rgba(255,255,255,.15);
    backdrop-filter:blur(12px);
    transition:.35s;
    position:relative;
    overflow:hidden;
}

.card:hover{
    transform:translateY(-8px);
    box-shadow:0 18px 35px rgba(0,0,0,.35);
}

.card::before{
    content:"";
    position:absolute;
    width:140px;
    height:140px;
    border-radius:50%;
    background:rgba(0,229,255,.08);
    top:-40px;
    right:-40px;
}

.card h2{
    margin-bottom:14px;
    font-size:28px;
}

.card p{
    color:#d5ddf7;
    line-height:1.7;
}

/* =====================================================
   TIMELINE / COLLABORATION
===================================================== */
.collab{
    max-width:1100px;
    margin:auto;
    text-align:center;
}

.collab h2{
    font-size:44px;
    margin-bottom:18px;
}

.collab p{
    font-size:18px;
    color:#dce7ff;
    line-height:1.8;
}

.flow{
    margin-top:35px;
    display:flex;
    flex-wrap:wrap;
    gap:18px;
    justify-content:center;
}

.node{
    padding:16px 22px;
    border-radius:16px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.18);
    min-width:180px;
    animation:glowNode 2.5s infinite alternate;
}

@keyframes glowNode{
    from{box-shadow:0 0 0 rgba(0,255,255,0);}
    to{box-shadow:0 0 18px rgba(0,255,255,.28);}
}

/* =====================================================
   FOOTER
===================================================== */
footer{
    text-align:center;
    padding:35px;
    color:#9fb0d1;
}

@media(max-width:768px){
    .hero h1{font-size:42px;}
    .hero p{font-size:17px;}
}
</style>
</head>
<body>

<!-- SUPERPOSITION PARTICLES -->
<div class="particle p1"></div>
<div class="particle p2"></div>
<div class="particle p3"></div>
<div class="particle p4"></div>
<div class="particle p5"></div>
<div class="particle p6"></div>

<!-- HERO -->
<section class="hero">
<div class="glass">
    <h1>Quantum AI Nexus</h1>
    <p>
        A futuristic convergence of Artificial Intelligence, Quantum Computing,
        Quantum Sensing, and Quantum Communication.
        Systems that learn faster, sense finer, and communicate securely.
    </p>

    <a href="#explore" class="btn">Enter the Field</a>

    <div class="eye-wrap">
        <div class="eye"></div>
    </div>
</div>
</section>

<div class="beam"></div>

<!-- CONCEPTS -->
<section class="section" id="explore">
<div class="grid">

<div class="card">
<h2>Artificial Intelligence</h2>
<p>
Learns patterns, reasons over data, automates decisions,
powers robotics, medical diagnostics, language systems,
and adaptive infrastructure.
</p>
</div>

<div class="card">
<h2>Quantum Computing</h2>
<p>
Uses qubits, superposition, and entanglement to explore
many states simultaneously, accelerating select classes
of optimization and simulation tasks.
</p>
</div>

<div class="card">
<h2>Quantum Sensing</h2>
<p>
Measures tiny changes in gravity, magnetic fields, time,
and motion with extreme precision for navigation, medicine,
earth science, and defense.
</p>
</div>

<div class="card">
<h2>Quantum Communication</h2>
<p>
Enables high-security information transfer using quantum
states and key distribution where interception becomes detectable.
</p>
</div>

</div>
</section>

<!-- COLLABORATION -->
<section class="section">
<div class="collab">
<h2>When Combined</h2>
<p>
AI can control quantum hardware, optimize experiments, decode sensor streams,
and orchestrate secure autonomous networks. Quantum processors can enhance
specific AI workloads, while quantum sensing supplies ultra-clean signals.
Together they form a strategic intelligence stack.
</p>

<div class="flow">
<div class="node">AI Reasoning</div>
<div class="node">Quantum Compute</div>
<div class="node">Quantum Sensors</div>
<div class="node">Secure Networks</div>
<div class="node">Autonomous Systems</div>
</div>
</div>
</section>

<!-- FUTURE -->
<section class="section">
<div class="grid">

<div class="card">
<h2>Healthcare</h2>
<p>
Molecular simulation, early disease detection,
real-time monitoring, personalized treatment systems.
</p>
</div>

<div class="card">
<h2>Climate & Energy</h2>
<p>
Better materials discovery, grid optimization,
environmental sensing, resilient logistics.
</p>
</div>

<div class="card">
<h2>Defense & Space</h2>
<p>
GPS-independent navigation, secure links,
advanced robotics, anomaly detection.
</p>
</div>

<div class="card">
<h2>Finance & Industry</h2>
<p>
Optimization, fraud defense, dynamic planning,
high-precision manufacturing control.
</p>
</div>

</div>
</section>

<footer>
Quantum AI Nexus • Experimental Landing Interface
</footer>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(PAGE)

if __name__ == "__main__":
    print("Quantum AI Landing Page Running...")
    print("Open: http://127.0.0.1:5000/")
    app.run(debug=True)