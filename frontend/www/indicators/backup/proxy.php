<?php
/**
 * API Proxy — procurement dashboard
 * In production, replace the dummy data below with real cURL calls
 * to the upstream REST API. The proxy also handles caching.
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

/* ---------- tiny file cache ---------- */
$cacheFile = sys_get_temp_dir() . '/procurement_cache.json';
$cacheTTL  = 900; // 15 minutes

if (file_exists($cacheFile) && (time() - filemtime($cacheFile)) < $cacheTTL) {
    echo file_get_contents($cacheFile);
    exit;
}

/* ---------- dummy data (replace with cURL to real API) ---------- */
function fetchFromAPI(): array {
    /*
     * PRODUCTION: replace body with something like:
     *
     * $ch = curl_init('https://api.example.com/procurement/indicators');
     * curl_setopt_array($ch, [
     *     CURLOPT_RETURNTRANSFER => true,
     *     CURLOPT_HTTPHEADER     => ['Authorization: Bearer ' . API_KEY],
     * ]);
     * $raw  = curl_exec($ch);
     * $data = json_decode($raw, true);
     * curl_close($ch);
     * return $data;
     */

    return [

        /* ── Indicator 1 ─────────────────────────────────────────── */
        'single_bidder' => [
            'id'          => 'single_bidder',
            'title'       => 'Single Bidder',
            'subtitle'    => 'Proportion of contracts awarded with only one bidder',
            'chart_type'  => 'donut_trend',   // hint for the front-end
            'by_lots' => [
                'total'              => 1540,
                'single_bidder_count'=> 620,
                'percentage'         => 40.26,
                'trend' => [
                    ['year' => 2014, 'percentage' => 30.1],
                    ['year' => 2015, 'percentage' => 31.4],
                    ['year' => 2016, 'percentage' => 29.8],
                    ['year' => 2017, 'percentage' => 33.2],
                    ['year' => 2018, 'percentage' => 35.0],
                    ['year' => 2019, 'percentage' => 36.7],
                    ['year' => 2020, 'percentage' => 37.5],
                    ['year' => 2021, 'percentage' => 38.1],
                    ['year' => 2022, 'percentage' => 39.0],
                    ['year' => 2023, 'percentage' => 40.26],
                ],
            ],
            'by_budget' => [
                'total_budget'          => 980000000,
                'single_bidder_budget'  => 340000000,
                'percentage'            => 34.69,
                'trend' => [
                    ['year' => 2014, 'percentage' => 27.1],
                    ['year' => 2015, 'percentage' => 28.3],
                    ['year' => 2016, 'percentage' => 26.9],
                    ['year' => 2017, 'percentage' => 29.4],
                    ['year' => 2018, 'percentage' => 30.8],
                    ['year' => 2019, 'percentage' => 31.5],
                    ['year' => 2020, 'percentage' => 32.0],
                    ['year' => 2021, 'percentage' => 33.1],
                    ['year' => 2022, 'percentage' => 33.9],
                    ['year' => 2023, 'percentage' => 34.69],
                ],
            ],
        ],

        /* ── Indicator 2 ─────────────────────────────────────────── */
        'direct_awards' => [
            'id'         => 'direct_awards',
            'title'      => 'Direct Awards',
            'subtitle'   => 'Procedures negotiated without open competition',
            'chart_type' => 'hbar',
            'available'  => true,
            'substitute_metric' => 'invitation_limit',
            'total'      => 1540,
            'by_invitation_limit' => [
                ['category' => 'Open',        'count' => 890, 'percentage' => 57.8],
                ['category' => 'Restricted',  'count' => 410, 'percentage' => 26.6],
                ['category' => 'Negotiated',  'count' => 177, 'percentage' => 11.5],
                ['category' => 'Direct',      'count' =>  63, 'percentage' =>  4.1],
            ],
        ],

        /* ── Indicator 3 ─────────────────────────────────────────── */
        'ted_publication' => [
            'id'          => 'ted_publication',
            'title'       => 'Publication Value',
            'subtitle'    => 'Tenders with a TED publication URL',
            'chart_type'  => 'area_trend',
            'total_tenders'       => 1540,
            'with_ted_url'        => 612,
            'without_ted_url'     => 928,
            'percentage_published'=> 39.74,
            'trend' => [
                ['period' => '2022-Q1', 'percentage' => 34.2],
                ['period' => '2022-Q2', 'percentage' => 35.8],
                ['period' => '2022-Q3', 'percentage' => 36.5],
                ['period' => '2022-Q4', 'percentage' => 37.0],
                ['period' => '2023-Q1', 'percentage' => 37.2],
                ['period' => '2023-Q2', 'percentage' => 38.9],
                ['period' => '2023-Q3', 'percentage' => 40.1],
                ['period' => '2023-Q4', 'percentage' => 39.74],
            ],
        ],

        /* ── Indicator 4 ─────────────────────────────────────────── */
        'joint_procurement' => [
            'id'       => 'joint_procurement',
            'title'    => 'Joint Procurement',
            'subtitle' => 'Contracts involving UTEs or subcontracting',
            'chart_type' => 'grouped_bar',
            'total'    => 1540,
            'ute_awards' => [
                'count'      => 187,
                'percentage' => 12.14,
            ],
            'with_subcontracting' => [
                'count'                   => 334,
                'percentage'              => 21.69,
                'avg_subcontracting_rate' => 28.4,
            ],
            'combined_joint' => [
                'count'      => 445,
                'percentage' => 28.9,
            ],
            'subcontracting_rate_distribution' => [
                ['range' => '1–10 %',  'count' =>  89],
                ['range' => '10–25 %', 'count' => 134],
                ['range' => '25–50 %', 'count' =>  78],
                ['range' => '50 %+',   'count' =>  33],
            ],
        ],

    ];
}

/* ---------- build response ---------- */
$data = [
    'as_of'      => date('Y-m-d\TH:i:s\Z'),
    'indicators' => fetchFromAPI(),
];

$json = json_encode($data, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);

// write cache
file_put_contents($cacheFile, $json);

echo $json;
