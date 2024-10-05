use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::collections::HashSet;
use indicatif::ProgressBar;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn filter_category_map(mut cat_to_doc: HashMap<String, Vec<String>>, missed_docs: HashSet<String>, missed_cats: Vec<String>) -> Py<PyAny> {
    let pb: ProgressBar = ProgressBar::new(missed_cats.len() as u64);

    for key in missed_cats {
        if let Some(docs) = cat_to_doc.get(&key).cloned() {
            cat_to_doc.insert(key, docs.into_iter().filter(|doc: &_| missed_docs.contains(doc)).collect());
        }
        pb.inc(1);
    }

    return Python::with_gil(|py: Python| {
        cat_to_doc.to_object(py)
    });
}

// #[pyfunction]
// fn explore_states(options: Vec<PyTuple>, visited: Vec<HashSet<String>>, sorted_categories: Vec<String>, current_solution: Vec<String>, cat_to_doc: HashMap<String, Vec<String>>, missed_docs: HashSet<String>, document_coverage: usize, available_categories: Vec<String>) -> Py<PyAny> {
//     let pb: ProgressBar = ProgressBar::new(sorted_categories.len() as u64);

//     for category in sorted_categories {
//         // Initialize new solution
//         let mut new_solution = current_solution.clone();
//         new_solution.push(category);

//         // Compute the new solution's document coverage
//         let mut covered_documents: HashSet<&str> = HashSet::new();
//         for solution_category in &new_solution {
//             if let Some(docs) = cat_to_doc.get(solution_category) {
//                 for doc in docs {
//                     if missed_docs.contains(doc) {
//                         covered_documents.insert(doc);
//                     }
//                 }
//             }
//         }

//         let new_document_coverage = covered_documents.len();

//         // Skip appending states that do not increase the coverage
//         if new_document_coverage <= document_coverage {
//             pb.inc(1);
//             continue;
//         }

//         // Remove the current category from the list of available categories
//         let mut remaining_categories = available_categories.clone();
//         remaining_categories.retain(|&c| c != category);

//         // Create new state tuple (remaining_categories, new_document_coverage, new_solution)
//         let new_state = (remaining_categories, new_document_coverage, new_solution);

//         // Update document coverage to reflect the new one
//         document_coverage = new_document_coverage;

//         pb.inc(1);
//     }

//     return Python::with_gil(|py: Python| {
//         options.to_vec()
//     });
// }

/// A Python module implemented in Rust.
#[pymodule]
fn rust_search_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_category_map, m)?)?;
    Ok(())
}
