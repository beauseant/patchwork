<?php include 'includes/header.php'; ?>
<?php include 'includes/sidebar.php'; #  eeH4jeka7ooR-u.soh?>

    <!-- Contenido principal -->
        <main class="main-content p-4">
          
            <div class="hero-section-inner d-flex justify-content-center align-items-center rounded mb-5">
            <div class="text-center p-3">
                <h1 class="display-4 fw-bold">Welcome to pAtChWoRK</h1>
                <p class="lead">An NLP-powered system that automates the extraction, enrichment, and exploitation of public procurement data to support stakeholders across the tendering process.</p>
            </div>
            </div>

            

            <div class="row text-center g-4 justify-content-center">
                <div class="col-md-6 col-lg mb-3 card-title">                    
                        <a href="/extract/load.php" class="text-decoration-none">
                        <div class="p-4 shadow-sm rounded bg-light h-100">                        
                            <i class="bi bi-cloud-upload fs-1 text-primary"></i>
                            <h3 class="mt-3 h5">Upload Procurement Documents</h3>
                            <p class="text-muted small">Upload Spanish procurement PDFs (technical or administrative) to automatically extract key metadata such as objectives, CPV codes, award criteria, and special conditions.</p>
                        </div>
                    </a>
                </div>

                <div class="col-md-6 col-lg mb-3 card-title">                        <a href="/archive/index.php" class="text-decoration-none">
                            <div class="p-4 shadow-sm rounded bg-light h-100">
                                <i class="bi bi-file-earmark-bar-graph fs-1 text-success"></i>
                                <h3 class="mt-3 h5">Explore Pre-Enriched Documents </h3>
                                <p class="text-muted small">Access already processed procurement files enriched with structured metadata for deeper analysis and retrieval.</p>
                            </div>
                        </a>
                </div>

                <div class="col-md-6 col-lg mb-3 card-title">                    <a href="/topics/index.php" class="text-decoration-none">
                        <div class="p-4 shadow-sm rounded bg-light h-100">
                            <i class="bi bi-columns fs-1 text-info"></i>
                            <h3 class="mt-3 h5">Discover Topics & Semantic Links</h3>
                            <p class="text-muted small">Explore topic models and semantic graphs to identify similar tenders and uncover thematic patterns across procurement data.</p>
                        </div>
                    </a>
                </div>

                <div class="col-md-6 col-lg mb-3 card-title">                    <a href="/semantic/index.php" class="text-decoration-none">
                        <div class="p-4 shadow-sm rounded bg-light h-100">
                            <i class="bi bi-intersect fs-1 text-secondary"></i>
                            <h3 class="mt-3 h5">Semantic & Thematic Similarity Search</h3>
                            <p class="text-muted small">Retrieve tenders with high semantic similarity or shared topics from our catalog.</p>
                        </div>                        
                    </a>
                </div>

                <div class="col-md-6 col-lg mb-3 card-title">                     <a href="/indicators/index.php" class="text-decoration-none">
                         <div class="p-4 shadow-sm rounded bg-light h-100">
                             <i class="bi bi-app-indicator fs-1 text-warning"></i>
                             <h3 class="mt-3 h5">Procurement Indicators</h3>
                             <p class="text-muted small">Visualize and analyze procurement metrics aligned with indicators for transparency and performance assessment.</p>
                         </div>
                     </a>
                 </div>

            </div>
            
            <div class="mt-5 text-muted ">
                <h4 class="" >About Us</h4>
                <p>We are researchers from Universidad Carlos III de Madrid, Universidad Politécnica de Madrid, and the City Council of Zaragoza working to make public procurement more transparent and efficient.</p>
                <p>pAtChWoRK builds on components developed in the NextProcurement Project (Open Harmonized and Enriched Public Procurement Platform, European Commission).  It uses NLP and large language models to extract, enrich, and explore procurement data, supporting administrators, auditors, and bidders to find and analyze information through semantic search, contract classification, and policy insights.</p>
                <p>Built for Spanish procurement data, pAtChWoRK’s modular design can easily adapt to other countries and contexts.</p>
                <p><b>Contact</b>: jarenas[at]ing.uc3m.es</p>
            </div>

        </main>
</div>     <!--<div class="wrapper d-flex"> -->
<?php include 'includes/footer.php'; ?>
