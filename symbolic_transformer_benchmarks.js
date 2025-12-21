/**
 * Symbolic Transformer Structural Benchmarks
 * 
 * These tests verify that the model has learned FOL syntax rules,
 * not just pattern matching. They form the foundation for reasoning capabilities.
 */

const BENCHMARKS = {
  
  // ============================================================
  // CATEGORY 1: DETERMINISTIC STRUCTURE
  // These should be near 100% - the grammar requires it
  // ============================================================
  
  deterministic: {
    name: "Deterministic Structure",
    description: "Hard syntactic constraints - model should be ~100% confident",
    tests: [
      {
        input: "FORALL",
        expected: "VAR",
        expectedMin: 95,
        rationale: "Universal quantifier must bind a variable"
      },
      {
        input: "EXISTS",
        expected: "VAR", 
        expectedMin: 95,
        rationale: "Existential quantifier must bind a variable"
      },
      {
        input: "PRED 3",
        expected: "LPAREN",
        expectedMin: 95,
        rationale: "Predicate with index must open argument list"
      },
      {
        input: "PRED 0 LPAREN VAR 0 COMMA VAR 1",
        expected: "RPAREN",
        expectedMin: 90,
        rationale: "Binary predicate with two arguments must close"
      },
      {
        input: "FUNC 2",
        expected: "LPAREN",
        expectedMin: 90,
        rationale: "Function with index must open argument list"
      }
    ]
  },

  // ============================================================
  // CATEGORY 2: CALIBRATED UNCERTAINTY
  // Model should show appropriate uncertainty across valid options
  // ============================================================
  
  calibratedUncertainty: {
    name: "Calibrated Uncertainty", 
    description: "Grammar branches here - model should distribute probability across valid continuations",
    tests: [
      {
        input: "NOT",
        expectedOneOf: ["PRED", "LPAREN", "FORALL", "EXISTS"],
        expectedMax: 40,
        rationale: "After NOT, multiple valid continuations: predicate, grouped formula, or quantified formula"
      },
      {
        input: "FORALL VAR 1",
        expectedOneOf: ["LPAREN", "PRED", "FORALL", "EXISTS", "NOT"],
        expectedMax: 50,
        rationale: "After binding a variable, can start body: predicate, group, nested quantifier, negation"
      },
      {
        input: "FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN",
        expectedOneOf: ["AND", "OR", "IMPLIES", "IFF"],
        expectedMax: 40,
        rationale: "Complete formula can extend with any connective"
      },
      {
        input: "LPAREN PRED 1 LPAREN VAR 0 RPAREN",
        expectedOneOf: ["AND", "OR", "IMPLIES", "IFF"],
        expectedMax: 35,
        rationale: "After first clause in compound, need binary connective"
      }
    ]
  },

  // ============================================================
  // CATEGORY 3: PREDICATE ARITY
  // Model should learn distribution of unary vs binary predicates
  // ============================================================
  
  predicateArity: {
    name: "Predicate Arity Distribution",
    description: "Model should learn that predicates can be unary or binary",
    tests: [
      {
        input: "PRED 7 LPAREN VAR 2",
        expectedOneOf: ["RPAREN", "COMMA"],
        expectedDistribution: { "RPAREN": [50, 80], "COMMA": [20, 50] },
        rationale: "After first argument: close (unary) or continue (binary)"
      },
      {
        input: "PRED 0 LPAREN VAR 0",
        expectedOneOf: ["RPAREN", "COMMA"],
        rationale: "Same test with different indices - should show similar distribution"
      },
      {
        input: "PRED 99 LPAREN VAR 50",
        expectedOneOf: ["RPAREN", "COMMA"],
        rationale: "Compositional test - large indices should behave same as small"
      }
    ]
  },

  // ============================================================
  // CATEGORY 4: PARENTHESIS TRACKING
  // Critical test: does the model track structural depth?
  // ============================================================
  
  parenthesisTracking: {
    name: "Parenthesis Balance Tracking",
    description: "Model must track open/close parens across sequence - tests structural memory",
    tests: [
      {
        input: "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1",
        expected: "RPAREN",
        expectedMin: 55,
        rationale: "Inside PRED 1's argument list - must close predicate paren first"
      },
      {
        input: "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
        expectedOneOf: ["RPAREN", "AND", "OR", "IMPLIES", "IFF"],
        rparenMin: 25,
        rationale: "Compound complete, one outer LPAREN unclosed - can close or extend"
      },
      {
        input: "LPAREN LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN RPAREN",
        expectedOneOf: ["RPAREN", "AND", "OR", "IMPLIES", "IFF"],
        rparenMin: 20,
        rationale: "Two levels of nesting - still one unclosed outer paren"
      },
      {
        input: "PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
        expectedOneOf: ["AND", "OR", "IMPLIES", "IFF"],
        rparenMax: 15,
        rationale: "All parens balanced - RPAREN should be LOW probability"
      }
    ]
  },

  // ============================================================
  // CATEGORY 5: COMPOSITIONAL GENERALISATION  
  // Same structure, different indices - tests true learning vs memorisation
  // ============================================================
  
  compositionalGeneralisation: {
    name: "Compositional Generalisation",
    description: "Model should treat VAR 0 and VAR 624 identically in structural context",
    tests: [
      {
        input: "FORALL VAR 0",
        compareWith: "FORALL VAR 99",
        rationale: "Different variable index should yield same structural predictions"
      },
      {
        input: "FORALL VAR 0",
        compareWith: "FORALL VAR 624",
        rationale: "Max single-digit base-625 index should behave identically"
      },
      {
        input: "PRED 0 LPAREN VAR 0",
        compareWith: "PRED 50 LPAREN VAR 100",
        rationale: "Different pred and var indices should yield same arity distribution"
      },
      {
        input: "EXISTS VAR 1 PRED 1 LPAREN VAR 1 RPAREN",
        compareWith: "EXISTS VAR 500 PRED 200 LPAREN VAR 500 RPAREN",
        rationale: "Complete formulas with different indices should have same continuation distribution"
      }
    ]
  },

  // ============================================================
  // CATEGORY 6: NESTED QUANTIFIERS
  // Tests understanding of scope and nesting
  // ============================================================
  
  nestedQuantifiers: {
    name: "Nested Quantifier Handling",
    description: "Model should handle multiple levels of quantification",
    tests: [
      {
        input: "FORALL VAR 0 FORALL VAR 1",
        expectedOneOf: ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
        rationale: "After two quantifiers, start body - same options as after one"
      },
      {
        input: "FORALL VAR 0 EXISTS VAR 1",
        expectedOneOf: ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
        rationale: "Mixed quantifiers should behave same as repeated"
      },
      {
        input: "FORALL VAR 0 EXISTS VAR 1 FORALL VAR 2",
        expectedOneOf: ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
        rationale: "Triple nesting - should still understand body comes next"
      }
    ]
  },

  // ============================================================
  // CATEGORY 7: IMPLICATION STRUCTURE
  // Critical for reasoning - model must understand implies
  // ============================================================
  
  implicationStructure: {
    name: "Implication Structure",
    description: "Understanding of antecedent → consequent structure",
    tests: [
      {
        input: "LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES",
        expectedOneOf: ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
        rationale: "RHS of implication needs a formula"
      },
      {
        input: "FORALL VAR 0 LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES PRED 1 LPAREN VAR 0 RPAREN",
        expected: "RPAREN",
        expectedMin: 30,
        rationale: "Complete implication inside quantifier scope - should close"
      }
    ]
  },

  // ============================================================
  // CATEGORY 8: ERROR RECOVERY / EDGE CASES
  // What does the model do with unusual inputs?
  // ============================================================
  
  edgeCases: {
    name: "Edge Cases",
    description: "Unusual or malformed inputs - observational, not pass/fail",
    tests: [
      {
        input: "RPAREN",
        observational: true,
        rationale: "Starting with close paren - what does model predict?"
      },
      {
        input: "AND",
        observational: true,
        rationale: "Starting with connective - what does model predict?"
      },
      {
        input: "VAR",
        observational: true,
        rationale: "Incomplete compositional token - how does model handle?"
      },
      {
        input: "LPAREN RPAREN",
        observational: true,
        rationale: "Empty parens - what continuation does model expect?"
      }
    ]
  }
};

// ============================================================
// BENCHMARK RUNNER
// ============================================================

class BenchmarkRunner {
  constructor(modelPredictor) {
    this.predictor = modelPredictor;
    this.results = {};
  }

  async runAll() {
    const report = {
      timestamp: new Date().toISOString(),
      categories: {},
      summary: { passed: 0, failed: 0, observational: 0 }
    };

    for (const [categoryKey, category] of Object.entries(BENCHMARKS)) {
      report.categories[categoryKey] = {
        name: category.name,
        description: category.description,
        tests: []
      };

      for (const test of category.tests) {
        const result = await this.runSingleTest(test);
        report.categories[categoryKey].tests.push(result);
        
        if (result.observational) {
          report.summary.observational++;
        } else if (result.passed) {
          report.summary.passed++;
        } else {
          report.summary.failed++;
        }
      }
    }

    report.summary.total = report.summary.passed + report.summary.failed + report.summary.observational;
    report.summary.passRate = (report.summary.passed / (report.summary.passed + report.summary.failed) * 100).toFixed(1);
    
    return report;
  }

  async runSingleTest(test) {
    const prediction = await this.predictor(test.input);
    
    const result = {
      input: test.input,
      rationale: test.rationale,
      prediction: prediction,
      observational: test.observational || false
    };

    if (test.observational) {
      result.note = "Observational test - no pass/fail criteria";
      return result;
    }

    // Check deterministic expectation
    if (test.expected && test.expectedMin) {
      const prob = prediction[test.expected] || 0;
      result.passed = prob >= test.expectedMin;
      result.detail = `Expected ${test.expected} >= ${test.expectedMin}%, got ${prob.toFixed(1)}%`;
    }
    
    // Check distribution expectation
    else if (test.expectedOneOf) {
      const probs = test.expectedOneOf.map(token => prediction[token] || 0);
      const total = probs.reduce((a, b) => a + b, 0);
      result.passed = total >= 50; // At least 50% in expected tokens
      result.detail = `Expected one of [${test.expectedOneOf.join(', ')}], total prob: ${total.toFixed(1)}%`;
      
      // Additional checks
      if (test.expectedMax) {
        const maxProb = Math.max(...probs);
        result.passed = result.passed && maxProb <= test.expectedMax;
        result.detail += ` (max should be <= ${test.expectedMax}%, got ${maxProb.toFixed(1)}%)`;
      }
      if (test.rparenMin && prediction['RPAREN']) {
        result.passed = result.passed && prediction['RPAREN'] >= test.rparenMin;
        result.detail += ` (RPAREN should be >= ${test.rparenMin}%, got ${prediction['RPAREN'].toFixed(1)}%)`;
      }
      if (test.rparenMax && prediction['RPAREN']) {
        result.passed = result.passed && prediction['RPAREN'] <= test.rparenMax;
        result.detail += ` (RPAREN should be <= ${test.rparenMax}%, got ${prediction['RPAREN'].toFixed(1)}%)`;
      }
    }
    
    // Compositional comparison
    else if (test.compareWith) {
      const prediction2 = await this.predictor(test.compareWith);
      result.comparedWith = test.compareWith;
      result.prediction2 = prediction2;
      
      // Check if top-5 tokens are similar
      const top5_1 = Object.entries(prediction).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
      const top5_2 = Object.entries(prediction2).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]);
      const overlap = top5_1.filter(t => top5_2.includes(t)).length;
      
      result.passed = overlap >= 4; // At least 4 of top 5 should match
      result.detail = `Top-5 overlap: ${overlap}/5 tokens match`;
    }

    return result;
  }

  generateReport(report) {
    let output = [];
    
    output.push("═".repeat(60));
    output.push("SYMBOLIC TRANSFORMER STRUCTURAL BENCHMARKS");
    output.push("═".repeat(60));
    output.push(`Timestamp: ${report.timestamp}`);
    output.push(`Pass Rate: ${report.summary.passRate}% (${report.summary.passed}/${report.summary.passed + report.summary.failed})`);
    output.push(`Observational: ${report.summary.observational}`);
    output.push("");

    for (const [key, category] of Object.entries(report.categories)) {
      output.push("─".repeat(60));
      output.push(`${category.name.toUpperCase()}`);
      output.push(`${category.description}`);
      output.push("─".repeat(60));
      
      for (const test of category.tests) {
        const status = test.observational ? "👁️" : (test.passed ? "✓" : "✗");
        output.push(`${status} Input: ${test.input}`);
        output.push(`  ${test.rationale}`);
        if (test.detail) {
          output.push(`  ${test.detail}`);
        }
        output.push("");
      }
    }

    return output.join("\n");
  }
}

// ============================================================
// EXPORT FOR USE IN DEMO
// ============================================================

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { BENCHMARKS, BenchmarkRunner };
}

// For browser use
if (typeof window !== 'undefined') {
  window.SymbolicBenchmarks = { BENCHMARKS, BenchmarkRunner };
}
