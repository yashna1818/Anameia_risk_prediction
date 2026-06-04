/* ==========================================================================
   ANEMIA RISK & NUTRIGENOMICS APP — FRONTEND LOGIC (VANILLA JS)
   ========================================================================== */

// Global State
let preloadedPatients = {};
let currentVcfData = null; // Stores parsed VCF details: { prs, prs_category, matched_snps, active_genes }
let activeGenotype = {};   // Current active SNPs mapping: { rsid: dosage }

// Gene Database for Explorer Tab (Instant Search)
const GENE_EXPLORER_DB = [
    { name: "TMPRSS6", pathway: "Hepcidin Regulation (Matriptase-2)", mechanism: "Suppresses hepcidin. Mutation causes elevated hepcidin, blocking iron absorption (IRIDA).", flag: true, badge: "Iron Lock" },
    { name: "HAMP", pathway: "Hepcidin Production (Master Hormone)", mechanism: "Encodes hepcidin. High expression degrades ferroportin and traps iron.", flag: true, badge: "Master Regulation" },
    { name: "CYBRD1", pathway: "Duodenal Iron Reduction (DcytB)", mechanism: "Reduces dietary Fe3+ to Fe2+. Defect impairs absorption, treatable with Vitamin C.", flag: false, badge: "Reduction Axis" },
    { name: "SLC11A2", pathway: "Intestinal Iron Uptake (DMT1)", mechanism: "Transports non-heme iron. Mutation impairs absorption; bypassable via heme iron.", flag: false, badge: "Transporter Axis" },
    { name: "HFE", pathway: "Iron Sensing & Homeostasis", mechanism: "Regulates hepcidin. Defect causes iron overload (hereditary hemochromatosis).", flag: true, badge: "Sensing Axis" },
    { name: "TF", pathway: "Blood Iron Transport (Transferrin)", mechanism: "Carries iron in blood. Variant reduces delivery rate to bone marrow.", flag: false, badge: "Transport Axis" },
    { name: "SLC40A1", pathway: "Cellular Iron Export (Ferroportin)", mechanism: "Exports iron. Mutation causes ferroportin disease and iron retention.", flag: true, badge: "Exporter Axis" },
    { name: "G6PD", pathway: "RBC Oxidative Protection (G6PD)", mechanism: "Deficiency causes hemolytic crisis when exposed to oxidants like fava beans.", flag: true, badge: "RBC Membrane" },
    { name: "PIEZO1", pathway: "RBC Volume Regulation", mechanism: "Dehydrates RBCs, causing fragility and splenic clearance (xerocytosis).", flag: false, badge: "RBC Membrane" },
    { name: "ACSL3", pathway: "RBC Lipid Activation", mechanism: "Incorporates fatty acids into membrane. Key determinant of RBC lifespan.", flag: false, badge: "Lipid Axis" },
    { name: "MTHFR", pathway: "Folate Metabolism (Methylation)", mechanism: "Converts folate to active form. Very common mutation, causes high homocysteine.", flag: false, badge: "Folate Axis" },
    { name: "ALAS2", pathway: "Haem Biosynthesis Step 1", mechanism: "B6-dependent enzyme. Mutation causes sideroblastic anemia (iron accumulation).", flag: true, badge: "Haem Axis" }
];

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function initApp() {
    setupTabNavigation();
    setupSliders();
    setupVcfDropzone();
    setupGeneExplorer();
    fetchModelMetrics();
    fetchPreloadedTemplates();

    // Attach major event listeners
    document.getElementById("btn-train-now").addEventListener("click", triggerRetrain);
    document.getElementById("btn-quick-train").addEventListener("click", triggerRetrain);
    document.getElementById("btn-run-assessment").addEventListener("click", runAssessment);
    document.getElementById("btn-start-assess").addEventListener("click", () => switchTab("assessment"));
    setupAdvisor();
}

// -----------------------------------------------------------------------------
// TAB NAVIGATION
// -----------------------------------------------------------------------------
function setupTabNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabId = item.getAttribute("data-tab");
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    // Update active nav button
    document.querySelectorAll(".nav-item").forEach(btn => {
        if (btn.getAttribute("data-tab") === tabId) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    // Update content panels
    document.querySelectorAll(".content-panel").forEach(panel => {
        if (panel.id === `panel-${tabId}`) {
            panel.classList.add("active");
        } else {
            panel.classList.remove("active");
        }
    });

    const titles = {
        home: { title: "Dashboard Home", subtitle: "Welcome to the Anemia Risk Prediction & Precision Nutrition Platform" },
        training: { title: "XGBoost Training Center", subtitle: "Evaluate model performance, check overfitting, and verify data balance" },
        assessment: { title: "Precision Advisor", subtitle: "Perform multi-omic diagnostics: genetic and phenotype assessment & AI nutrition plans" },
        explorer: { title: "Nutrigenomics Pathway Explorer", subtitle: "Browse biological explanations and gene-nutrient interactions" }
    };

    if (titles[tabId]) {
        document.getElementById("page-title").textContent = titles[tabId].title;
        document.getElementById("page-subtitle").textContent = titles[tabId].subtitle;
    }
    
    // Scroll content to top
    document.querySelector(".main-content").scrollTop = 0;
}

// -----------------------------------------------------------------------------
// SLIDER HANDLERS
function setupSliders() {
    // Sliders removed in consolidated dashboard layout
}

// -----------------------------------------------------------------------------
// GENE EXPLORER SEARCH & RENDER
// -----------------------------------------------------------------------------
function setupGeneExplorer() {
    renderGeneCards(GENE_EXPLORER_DB);

    const searchInput = document.getElementById("explorer-search-input");
    searchInput.addEventListener("input", (e) => {
        const term = e.target.value.toLowerCase().trim();
        const filtered = GENE_EXPLORER_DB.filter(g => 
            g.name.toLowerCase().includes(term) || 
            g.pathway.toLowerCase().includes(term) ||
            g.mechanism.toLowerCase().includes(term)
        );
        renderGeneCards(filtered);
    });
}

function renderGeneCards(genesList) {
    const grid = document.getElementById("explorer-genes-grid");
    grid.innerHTML = "";

    if (genesList.length === 0) {
        grid.innerHTML = `<div class="col-span-2 text-center text-secondary py-4">No matching genes found in the database.</div>`;
        return;
    }

    genesList.forEach(g => {
        const card = document.createElement("div");
        card.className = "gene-card";
        card.innerHTML = `
            <div class="gene-header">
                <span class="gene-name">${g.name}</span>
                <span class="gene-badge" style="background-color: ${g.flag ? 'rgba(255, 0, 127, 0.15)' : 'rgba(0, 242, 254, 0.15)'}; color: ${g.flag ? 'var(--accent-red)' : 'var(--accent-cyan)'};">${g.badge}</span>
            </div>
            <div class="gene-pathway-title">${g.pathway}</div>
            <p class="gene-mechanism">${g.mechanism}</p>
            <div class="gene-footer">
                <span class="gene-flag">${g.flag ? '<i class="fa-solid fa-triangle-exclamation"></i> Clinical Flag Active' : 'General Variant'}</span>
                <span class="text-cyan text-xs">Explore Recommendations <i class="fa-solid fa-chevron-right"></i></span>
            </div>
        `;

        card.addEventListener("click", () => {
            // Fill manual SNP values if relevant, and switch to assessment tab
            const snpMap = {
                "TMPRSS6": "rs2251655",
                "ACSL3": "rs6762719",
                "PIEZO1": "rs551118",
                "FADS1": "rs174568"
            };
            if (snpMap[g.name]) {
                const select = document.querySelector(`.snp-select[data-snp="${snpMap[g.name]}"]`);
                if (select) {
                    select.value = "2"; // set to hom
                    select.dispatchEvent(new Event("change"));
                }
                switchTab("assessment");
                // Scroll down to the manual SNP selector
                document.getElementById("manual-snps-container").scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(`Gene ${g.name} details: ${g.mechanism} Recommendations: Increase dietary intake of support nutrients and pair iron foods with Vitamin C.`);
            }
        });

        grid.appendChild(card);
    });
}

// -----------------------------------------------------------------------------
// VCF UPLOAD DRAG-AND-DROP
// -----------------------------------------------------------------------------
function setupVcfDropzone() {
    const dropzone = document.getElementById("vcf-dropzone");
    const fileInput = document.getElementById("vcf-file-input");
    const removeBtn = document.getElementById("btn-remove-vcf");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleVcfFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleVcfFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearVcf();
    });
}

function handleVcfFile(file) {
    const dropzone = document.getElementById("vcf-dropzone");
    const statusBox = document.getElementById("vcf-status-box");
    const nameLbl = document.getElementById("vcf-name-lbl");

    nameLbl.textContent = file.name;
    dropzone.classList.add("hidden");
    statusBox.classList.remove("hidden");

    // Upload and parse VCF via API
    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("vcf-snps-matched").textContent = "Parsing file...";

    fetch("/api/parse-vcf", {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) throw new Error("VCF parsing failed.");
        return res.json();
    })
    .then(data => {
        currentVcfData = data;
        
        // Update VCF Status displays
        document.getElementById("vcf-snps-matched").textContent = `${data.matched_count} / ${data.total_significant_snps}`;
        document.getElementById("vcf-prs-val").textContent = data.prs.toFixed(4);
        
        const catLbl = document.getElementById("vcf-prs-cat");
        catLbl.textContent = data.prs_category;
        catLbl.className = `text-${data.prs_category === 'High' ? 'red' : data.prs_category === 'Moderate' ? 'orange' : 'green'}`;

        // Map VCF genotypes into activeGenotype
        activeGenotype = {};
        data.matched_snps.forEach(item => {
            activeGenotype[item.snp] = item.dosage;
        });

        // Hide manual SNPs panel since VCF is active
        document.getElementById("manual-snps-container").classList.add("hidden");
    })
    .catch(err => {
        alert("Error parsing VCF: Make sure it is a valid VCF format (.vcf or compressed .vcf.gz).");
        clearVcf();
    });
}

function clearVcf() {
    currentVcfData = null;
    activeGenotype = {};
    document.getElementById("vcf-file-input").value = "";
    document.getElementById("vcf-dropzone").classList.remove("hidden");
    document.getElementById("vcf-status-box").classList.add("hidden");
    document.getElementById("manual-snps-container").classList.remove("hidden");
}

// -----------------------------------------------------------------------------
// PRELOADED PATIENTS & TEMPLATES
// -----------------------------------------------------------------------------
function fetchPreloadedTemplates() {
    fetch("/api/preloaded-patients")
    .then(res => res.json())
    .then(data => {
        preloadedPatients = data;
        renderTemplateButtons(data);
    })
    .catch(err => console.error("Error loading templates", err));
}

function renderTemplateButtons(templates) {
    const container = document.getElementById("patient-templates");
    container.innerHTML = "";

    Object.keys(templates).forEach(key => {
        const patient = templates[key];
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "template-btn";
        btn.textContent = patient.label;
        
        btn.addEventListener("click", () => {
            // Remove active classes
            document.querySelectorAll(".template-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Clear VCF if any was loaded
            clearVcf();

            // Populate Phenotype sliders & fields
            const pheno = patient.phenotype;
            document.getElementById("RIAGENDR").value = pheno.RIAGENDR;
            document.getElementById("RIDAGEYR").value = pheno.RIDAGEYR;
            document.getElementById("BMXWT").value = pheno.BMXWT;
            document.getElementById("BMXHT").value = pheno.BMXHT;
            document.getElementById("BMXWAIST").value = pheno.BMXWAIST;
            document.getElementById("INDFMPIR").value = pheno.INDFMPIR;
            document.getElementById("DIET_PREF").value = patient.diet_pref || "Any";
            document.getElementById("CUISINE_PREF").value = patient.cuisine_pref || "Western";

            // Sliders
            const sliders = ["DR1TIRON", "DR1TVC", "DR1TCALC", "DR1TPROT", "DR1TKCAL"];
            sliders.forEach(s => {
                const slider = document.getElementById(s);
                slider.value = pheno[s];
                slider.dispatchEvent(new Event("input"));
            });

            // Populate Manual SNPs
            const snpSelects = document.querySelectorAll(".snp-select");
            snpSelects.forEach(select => {
                const snpId = select.getAttribute("data-snp");
                if (patient.snps[snpId] !== undefined) {
                    select.value = patient.snps[snpId];
                } else {
                    select.value = "0";
                }
            });

            // Automatically run prediction assessment for the template
            runAssessment();
        });

        container.appendChild(btn);
    });
}

// -----------------------------------------------------------------------------
// MODEL METRICS
// -----------------------------------------------------------------------------
function fetchModelMetrics() {
    // In our case, the backend trains models automatically on launch.
    // Fetching the logs is simple, we will print them.
    // We can also fetch the metrics from a quick request
}

function triggerRetrain() {
    const btn = document.getElementById("btn-train-now");
    const spinner = document.getElementById("train-spinner");
    btn.disabled = true;
    spinner.classList.remove("hidden");

    fetch("/api/train", { method: "POST" })
    .then(res => {
        if (!res.ok) throw new Error("Retraining failed.");
        return res.json();
    })
    .then(data => {
        // Update Classifier Metrics
        document.getElementById("clf-accuracy").textContent = data.classifier.accuracy.toFixed(4);
        document.getElementById("clf-auc").textContent = data.classifier.test_auc.toFixed(4);
        document.getElementById("clf-f1").textContent = data.classifier.f1.toFixed(4);
        document.getElementById("clf-recall").textContent = data.classifier.recall.toFixed(4);
        document.getElementById("clf-train-auc").textContent = data.classifier.train_auc.toFixed(4);
        document.getElementById("clf-test-auc").textContent = data.classifier.test_auc.toFixed(4);
        
        const clfGap = Math.abs(data.classifier.train_auc - data.classifier.test_auc);
        document.getElementById("clf-auc-gap").textContent = clfGap.toFixed(4);
        
        const clfStatus = document.getElementById("clf-overfit-status");
        if (clfGap > 0.08) {
            clfStatus.className = "badge badge-warning";
            clfStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Mild Overfitting Detected';
        } else {
            clfStatus.className = "badge badge-success";
            clfStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> No significant overfitting';
        }

        // Update Regressor Metrics
        document.getElementById("reg-r2").textContent = data.regressor.test_r2.toFixed(4);
        document.getElementById("reg-mae").textContent = `${data.regressor.mae.toFixed(2)} g/dL`;
        document.getElementById("reg-rmse").textContent = `${data.regressor.rmse.toFixed(2)} g/dL`;
        document.getElementById("reg-iter").textContent = data.regressor.best_iteration;
        document.getElementById("reg-train-r2").textContent = data.regressor.train_r2.toFixed(4);
        document.getElementById("reg-test-r2").textContent = data.regressor.test_r2.toFixed(4);

        const regGap = Math.abs(data.regressor.train_r2 - data.regressor.test_r2);
        document.getElementById("reg-r2-gap").textContent = regGap.toFixed(4);

        const regStatus = document.getElementById("reg-overfit-status");
        if (regGap > 0.10) {
            regStatus.className = "badge badge-warning";
            regStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Mild Overfitting Detected';
        } else {
            regStatus.className = "badge badge-success";
            regStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> No significant overfitting';
        }

        // Append logs
        const logs = document.getElementById("training-logs");
        logs.textContent += `\n\n[${new Date().toLocaleTimeString()}] Retraining completed successfully. Integrated ${data.n_features} features, ${data.n_snps} SNPs.`;
        logs.scrollTop = logs.scrollHeight;

        alert("Model pipeline successfully retrained with leakage-free parameters!");
        switchTab("training");
    })
    .catch(err => {
        alert("Retraining failed: check server terminal logs.");
    })
    .finally(() => {
        btn.disabled = false;
        spinner.classList.add("hidden");
    });
}

// -----------------------------------------------------------------------------
// RUN DIAGNOSTICS & PREDICTION
// -----------------------------------------------------------------------------
function runAssessment() {
    const btn = document.getElementById("btn-run-assessment");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing multi-omic model...`;

    // 1. Gather Phenotype inputs with default maps replacing sliders & PIR
    const intakePattern = document.getElementById("adv-intake-pattern").value;
    const mostConsumed = document.getElementById("adv-most-consumed").value;
    
    // Default mappings for nutrients based on selected Eating Habit
    const defaultDietMap = {
        "Balanced Diet": { iron: 12.0, vitc: 60.0, calc: 900.0, prot: 65.0, kcal: 2000.0 },
        "High Carb Diet": { iron: 10.0, vitc: 50.0, calc: 800.0, prot: 50.0, kcal: 1800.0 },
        "High Protein Diet": { iron: 15.0, vitc: 80.0, calc: 1000.0, prot: 120.0, kcal: 2200.0 },
        "High Fat Diet": { iron: 9.0, vitc: 45.0, calc: 850.0, prot: 75.0, kcal: 2100.0 },
        "Low Carb Diet": { iron: 11.0, vitc: 70.0, calc: 950.0, prot: 90.0, kcal: 1700.0 }
    };
    
    const defaults = defaultDietMap[intakePattern] || defaultDietMap["Balanced Diet"];

    const phenotype = {
        RIAGENDR: parseInt(document.getElementById("RIAGENDR").value),
        RIDAGEYR: parseInt(document.getElementById("RIDAGEYR").value),
        BMXWT: parseFloat(document.getElementById("BMXWT").value),
        BMXHT: parseFloat(document.getElementById("BMXHT").value),
        BMXWAIST: parseFloat(document.getElementById("BMXWAIST").value),
        INDFMPIR: 2.5, // Default PIR
        BMXBMI: parseFloat((parseFloat(document.getElementById("BMXWT").value) / Math.pow(parseFloat(document.getElementById("BMXHT").value)/100, 2)).toFixed(1)),
        DR1TIRON: defaults.iron,
        DR1TVC: defaults.vitc,
        DR1TCALC: defaults.calc,
        DR1TPROT: defaults.prot,
        DR1TKCAL: defaults.kcal
    };

    // Calculate BMI inside input structure
    phenotype.BMXBMI = isNaN(phenotype.BMXBMI) ? 22.0 : phenotype.BMXBMI;

    // 2. Gather SNP values (either from VCF or manual selects)
    let snps = {};
    if (currentVcfData) {
        snps = activeGenotype;
    } else {
        const snpSelects = document.querySelectorAll(".snp-select");
        snpSelects.forEach(select => {
            snps[select.getAttribute("data-snp")] = parseInt(select.value);
        });
    }

    const diet_pref = document.getElementById("DIET_PREF").value;
    const cuisine_pref = document.getElementById("CUISINE_PREF").value;

    const checkedAllergies = [];
    document.querySelectorAll("input[name='adv-allergies']:checked").forEach(cb => {
        checkedAllergies.push(cb.value);
    });

    let dietPrefRec = "Non-Vegetarian";
    if (diet_pref === "Vegetarian") dietPrefRec = "Vegetarian";
    if (diet_pref === "Vegan") dietPrefRec = "Vegan";

    // 3. Post to prediction and advisor recommendation APIs in parallel
    Promise.all([
        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ phenotype, snps, diet_pref, cuisine_pref })
        }).then(res => {
            if (!res.ok) throw new Error("Anemia risk prediction failed");
            return res.json();
        }),
        fetch("/api/nutrition-recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                diet_pref: dietPrefRec,
                allergies: checkedAllergies,
                intake_pattern: intakePattern,
                most_consumed: mostConsumed
            })
        }).then(res => {
            if (!res.ok) throw new Error("AI Nutrition recommendation failed");
            return res.json();
        })
    ])
    .then(([predictData, recommendData]) => {
        renderPredictionReport(predictData);
        renderMealPlanner(predictData);
        renderAdvisorResults(recommendData);
        
        // Show report panel
        document.getElementById("report-section").classList.remove("hidden");
        document.getElementById("report-section").scrollIntoView({ behavior: 'smooth' });
    })
    .catch(err => {
        alert("Error executing prediction model: " + err.message);
    })
    .finally(() => {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-stethoscope"></i> Calculate Anemia Risk & Generate Diet`;
    });
}

function renderPredictionReport(data) {
    // 1. Risk Score Gauge Animation
    const scoreVal = data.anemia_risk_score;
    document.getElementById("report-score-val").textContent = scoreVal.toFixed(1);
    
    // Gauge stroke-dasharray is 251.2
    const dashoffset = 251.2 - (scoreVal / 100) * 251.2;
    document.getElementById("gauge-fill").style.strokeDashoffset = dashoffset;

    // 2. Risk Badge & stats
    const riskLbl = document.getElementById("report-risk-lbl");
    riskLbl.textContent = data.overall_risk;
    
    const riskDot = document.getElementById("report-risk-dot");
    const riskWrapper = document.getElementById("report-risk-wrapper");

    // Clear classes
    riskDot.className = "risk-pulse-dot";
    
    if (data.overall_risk === 'HIGH RISK') {
        riskDot.classList.add("high");
        riskLbl.className = "text-red";
    } else if (data.overall_risk === 'MODERATE RISK') {
        riskDot.classList.add("mod");
        riskLbl.className = "text-orange";
    } else {
        riskDot.classList.add("low");
        riskLbl.className = "text-green";
    }

    // predicted Hb and genetic PRS
    document.getElementById("report-hb-val").textContent = data.predicted_hb.toFixed(2);
    document.getElementById("report-prs-val-card").textContent = data.prs_value.toFixed(4);
    document.getElementById("report-prs-cat-lbl").textContent = data.prs_category;
    document.getElementById("report-prs-cat-lbl").className = `num text-${data.prs_category === 'High' ? 'red' : data.prs_category === 'Moderate' ? 'orange' : 'green'}`;
    
    document.getElementById("report-priority-text").textContent = `Priority Recommendation: ${data.priority}`;

    // 3. Pathway Burden Progress Bars
    const barsContainer = document.getElementById("report-pathway-bars");
    barsContainer.innerHTML = "";

    // Sort pathways descending
    const sortedPathways = Object.keys(data.pathway_burden_scores).map(key => {
        return { name: key, score: data.pathway_burden_scores[key] };
    }).sort((a, b) => b.score - a.score);

    sortedPathways.forEach(p => {
        const item = document.createElement("div");
        item.className = "pathway-bar-item";

        // Display names lookup mapping
        const nameMap = {
            hepcidin: "Hepcidin Regulation",
            absorption: "Duodenal Absorption",
            transferrin: "Blood Iron Transport (TF)",
            erythropoiesis: "Erythropoiesis Rate",
            folate_b12: "Folate & B12 Activation",
            rbc_membrane: "RBC Membrane Structure",
            rbc_energy: "RBC Purine Glycolysis",
            haem_synthesis: "Haem Assembly (B6)",
            iron_sensing: "Iron Overload Sensing",
            oxidative_protection: "Oxidative Membrane G6PD"
        };

        const pathName = nameMap[p.name] || p.name;
        const colorClass = p.score > 70 ? 'fill-red' : p.score > 30 ? 'fill-orange' : 'fill-cyan';

        item.innerHTML = `
            <span class="pathway-lbl">${pathName}</span>
            <div class="pathway-bar-track">
                <div class="pathway-bar-fill ${colorClass}" style="width: 0%;"></div>
            </div>
            <span class="pathway-val text-${p.score > 70 ? 'red' : p.score > 30 ? 'orange' : 'cyan'}">${p.score.toFixed(1)}%</span>
        `;
        barsContainer.appendChild(item);

        // Simple trigger for layout paint -> triggers sliding animation
        setTimeout(() => {
            item.querySelector(".pathway-bar-fill").style.width = `${p.score}%`;
        }, 100);
    });

    // 4. Clinical Referral Flags & Conflicts Alerts
    const alertsContainer = document.getElementById("report-alerts-container");
    const alertsBox = alertsContainer.querySelector("div");
    alertsBox.innerHTML = "";

    let hasAlerts = false;

    // Nutrient Conflict Alerts
    data.conflicts.forEach(c => {
        hasAlerts = true;
        const card = document.createElement("div");
        card.className = "alert-card alert-card-danger";
        card.innerHTML = `
            <div class="alert-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="alert-body">
                <h4>Nutritional Pathway Conflict Detected</h4>
                <p>${c}</p>
            </div>
        `;
        alertsBox.appendChild(card);
    });

    // Gene Clinical Flags
    data.clinical_flags.forEach(f => {
        hasAlerts = true;
        const card = document.createElement("div");
        card.className = "alert-card alert-card-warning";
        card.innerHTML = `
            <div class="alert-icon"><i class="fa-solid fa-circle-radiation"></i></div>
            <div class="alert-body">
                <h4>Gene Clinical Advisory</h4>
                <p>${f}</p>
            </div>
        `;
        alertsBox.appendChild(card);
    });

    if (hasAlerts) {
        alertsContainer.classList.remove("hidden");
    } else {
        alertsContainer.classList.add("hidden");
    }

    // 5. Explanations List
    const explContainer = document.getElementById("report-explanations-list");
    explContainer.innerHTML = "";

    if (data.pathway_explanations.length === 0) {
        explContainer.innerHTML = `<p class="text-secondary text-sm">No risk alleles detected. Standard homeostatic pathway operation active.</p>`;
    } else {
        data.pathway_explanations.forEach(e => {
            const row = document.createElement("div");
            row.className = "card bg-dark-glass mt-3";
            row.innerHTML = `
                <div class="card-body">
                    <div class="flex-between">
                        <strong class="text-cyan text-base">${e.gene} (Variant ${e.snp})</strong>
                        <span class="badge badge-success">Dosage: ${e.dosage} | Beta: ${e.beta.toFixed(3)}</span>
                    </div>
                    <div class="text-sm text-orange mt-2">Pathway: ${e.pathway}</div>
                    <p class="text-secondary text-sm mt-2"><strong>Molecular Mechanism:</strong> ${e.mechanism}</p>
                    <p class="text-secondary text-sm mt-1"><strong>Anemia Correlation:</strong> ${e.anemia_link}</p>
                    <p class="text-secondary text-sm mt-1"><strong>Nutritional Action:</strong> ${e.diet_impact}</p>
                </div>
            `;
            explContainer.appendChild(row);
        });
    }
}

function renderMealPlanner(data) {
    const meal = data.meal_plan;
    const arith = meal.arithmetic || {};

    // 1. Plan schedule details
    document.getElementById("meal-title").textContent = meal.title;
    document.getElementById("meal-breakfast").textContent = meal.breakfast;
    document.getElementById("meal-lunch").textContent = meal.lunch;
    document.getElementById("meal-dinner").textContent = meal.dinner;
    document.getElementById("meal-morningsnack").textContent = meal.snack || "Pumpkin Seeds (30g)";
    document.getElementById("meal-eveningsnack").textContent = arith.portion_scale > 1.2 
        ? `Double Portion Side Salad / Greens [Scale Factor: ${arith.portion_scale}x active]` 
        : "Walnuts & Almonds Mix (30g) + Water";

    // 2. Iron gap card details
    document.getElementById("meal-iron-gap").textContent = data.daily_iron_gap_mg.toFixed(1);
    document.getElementById("meal-rda-val").textContent = `${data.rda_mg} mg`;
    
    // Fill in nutrient arithmetic summary fields
    document.getElementById("summary-target-gap").textContent = `${arith.iron_gap_mg || 0} mg`;
    document.getElementById("summary-plan-iron").textContent = `${arith.plan_iron_mg || 0} mg`;
    document.getElementById("summary-plan-vitc").textContent = `${arith.plan_vit_c_mg || 0} mg`;
    document.getElementById("summary-plan-scaling").textContent = `${arith.portion_scale || 1.0}x`;

    // Dynamic genetic note
    const geneticNote = document.getElementById("meal-genetic-note");
    let hasSpecialPathways = false;
    data.active_genes.forEach(g => {
        if (["TMPRSS6", "CYBRD1", "MTHFR", "ALAS2"].includes(g.gene)) {
            hasSpecialPathways = true;
        }
    });
    if (hasSpecialPathways) {
        geneticNote.innerHTML = `🧬 <strong>Active Pathway Override:</strong> Enhanced Vitamin C pairings, absorption helpers, and B6/folate co-factors dynamically integrated based on genetic variants.`;
        geneticNote.style.color = "var(--accent-cyan)";
    } else {
        geneticNote.innerHTML = `🧬 <strong>Low Genetic Risk Profile:</strong> Standard nutritional guidelines applied. No genetic pathway burdens detected.`;
        geneticNote.style.color = "var(--text-secondary)";
    }

    // Estimate percent met
    const intake = data.rda_mg - data.daily_iron_gap_mg;
    const pct = Math.min(100, Math.max(5, (intake / data.rda_mg) * 100));
    const bar = document.getElementById("meal-iron-bar");
    bar.style.width = `${pct}%`;
    
    // Set color code for progress bar
    if (pct < 40) {
        bar.style.backgroundColor = "var(--accent-red)";
    } else if (pct < 80) {
        bar.style.backgroundColor = "var(--accent-orange)";
    } else {
        bar.style.backgroundColor = "var(--accent-green)";
    }

    // 3. Pathway override alert
    const overrideBox = document.getElementById("meal-override-alert");
    if (data.meal_override_reason.includes("none")) {
        overrideBox.classList.add("hidden");
    } else {
        overrideBox.classList.remove("hidden");
        document.getElementById("meal-override-reason").textContent = data.meal_override_reason;
    }

    // 4. Vitamin C boosters
    const boostersList = document.getElementById("meal-vitc-boosters");
    boostersList.innerHTML = "";
    data.vitamin_c_boosters.forEach(b => {
        const parts = b.split('—');
        const title = parts[0].split(':')[0].trim();
        const quantity = parts[0].split(':')[1].trim();
        const rationale = parts[1].trim();
        
        const li = document.createElement("li");
        li.innerHTML = `<strong>${title} (${quantity})</strong> ${rationale}`;
        boostersList.appendChild(li);
    });

    // 5. Foods to eat & avoid
    const eatList = document.getElementById("meal-foods-eat");
    eatList.innerHTML = "";
    
    // Base recommended foods
    data.base_foods_eat_more.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item.replace("Eat MORE ", "");
        eatList.appendChild(li);
    });

    // Gene specific custom foods
    Object.keys(data.gene_specific_foods).forEach(gene => {
        data.gene_specific_foods[gene].forEach(item => {
            const li = document.createElement("li");
            li.innerHTML = `<span class="text-cyan">🧬 Gene ${gene}:</span> ${item}`;
            eatList.appendChild(li);
        });
    });

    // Foods to avoid list
    const avoidList = document.getElementById("meal-foods-avoid");
    avoidList.innerHTML = "";
    data.foods_to_avoid.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        avoidList.appendChild(li);
    });
}

// -----------------------------------------------------------------------------
// AI NUTRITION ADVISOR CONTROLLERS
// -----------------------------------------------------------------------------
function setupAdvisor() {
    // Advisor logic consolidated into main runAssessment process
}

function renderAdvisorResults(data) {
    // 1. Render expected benefits
    const benefitsList = document.getElementById("report-benefits-list");
    benefitsList.innerHTML = "";
    if (data.nutritional_benefits && data.nutritional_benefits.length > 0) {
        data.nutritional_benefits.forEach(b => {
            const li = document.createElement("li");
            li.style.marginBottom = "6px";
            li.innerHTML = `<i class="fa-solid fa-shield-halved text-cyan" style="margin-right: 8px;"></i> ${b}`;
            benefitsList.appendChild(li);
        });
    } else {
        benefitsList.innerHTML = `<li class="text-secondary text-sm">No specific expected benefits returned.</li>`;
    }

    // 2. Render clinical advice
    const adviceBody = document.getElementById("report-clinical-advice-body");
    adviceBody.innerHTML = "";
    if (data.nutritional_advice && data.nutritional_advice.length > 0) {
        data.nutritional_advice.forEach(adv => {
            const div = document.createElement("div");
            div.style.marginTop = "10px";
            div.style.fontSize = "0.9rem";
            div.style.lineHeight = "1.4";
            div.style.color = "var(--text-primary)";
            div.style.display = "flex";
            div.style.gap = "8px";
            div.style.alignItems = "flex-start";
            div.innerHTML = `
                <i class="fa-solid fa-circle-info" style="color: var(--accent-blue); margin-top: 3px;"></i>
                <span>${adv}</span>
            `;
            adviceBody.appendChild(div);
        });
    } else {
        adviceBody.innerHTML = `<p class="text-secondary text-sm">No clinical advice available.</p>`;
    }

    // 3. Render advisor suggested meal plan outline
    const outlineBody = document.getElementById("report-advisor-meal-outline");
    outlineBody.innerHTML = "";
    if (data.suggested_meal_plan) {
        const meals = [
            { label: "Breakfast", color: "var(--accent-cyan)", text: data.suggested_meal_plan.breakfast },
            { label: "Lunch", color: "var(--accent-green)", text: data.suggested_meal_plan.lunch },
            { label: "Dinner", color: "var(--accent-orange)", text: data.suggested_meal_plan.dinner },
            { label: "Snacks", color: "#a55eea", text: data.suggested_meal_plan.snacks },
            { label: "Drinks", color: "#4b7bec", text: data.suggested_meal_plan.drinks }
        ];

        meals.forEach(m => {
            const itemDiv = document.createElement("div");
            itemDiv.style.display = "flex";
            itemDiv.style.gap = "10px";
            itemDiv.style.alignItems = "flex-start";
            itemDiv.style.borderBottom = "1px dashed rgba(255,255,255,0.05)";
            itemDiv.style.paddingBottom = "10px";
            itemDiv.style.marginTop = "10px";
            itemDiv.innerHTML = `
                <span style="background: ${m.color}; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; min-width: 85px; text-align: center; display: inline-block;">${m.label}</span>
                <div style="font-size: 0.85rem; color: var(--text-primary); flex: 1;">${m.text}</div>
            `;
            outlineBody.appendChild(itemDiv);
        });
    }
}
