use pyo3::prelude::*;
// use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::collections::HashSet;
use indicatif::ProgressBar;


fn filter_category_map_documents(
    cat_to_doc: &mut HashMap<String, Vec<String>>, 
    missed_docs: &HashSet<String>, 
    missed_cats: Vec<String>
) -> HashMap<String, Vec<String>> {
    // Initialize progress bar.
    let pb: ProgressBar = ProgressBar::new(missed_cats.len() as u64);

    // Initialize a new category to documents hashmap.
    let mut new_cat_to_doc: HashMap<String, Vec<String>> = HashMap::new();

    // Iterate through all categories in the missed categories vector.
    for key in missed_cats {
        if let Some(docs) = cat_to_doc.get(&key).cloned() {
            // If the category is a valid key in the category to 
            // documents hashmap, insert the documents (filtered to
            // only include documents from the missed document set) 
            // into the new hashmap under the same key. 
            new_cat_to_doc.insert(
                key, 
                docs.into_iter()
                    .filter(|doc: &_| missed_docs.contains(doc))
                    .collect()
            );
        }

        // Increment the progress bar.
        pb.inc(1);
    }

    // Return the new hashmap.
    return new_cat_to_doc
}

#[pyfunction]
fn filter_category_map(mut cat_to_doc: HashMap<String, Vec<String>>, missed_docs: HashSet<String>, missed_cats: Vec<String>) -> Py<PyAny> {
    // Initialize progress bar.
    let pb: ProgressBar = ProgressBar::new(missed_cats.len() as u64);

    // Iterate through all categories in the missed categories vector.
    for key in missed_cats {
        if let Some(docs) = cat_to_doc.get(&key).cloned() {
            // If 
            cat_to_doc.insert(
                key, 
                docs.into_iter()
                    .filter(|doc: &_| missed_docs.contains(doc))
                    .collect()
            );
        }

        // Increment the progress bar.
        pb.inc(1);
    }

    // Return the category to docs hashmap as a python object.
    return Python::with_gil(|py: Python| {
        cat_to_doc.to_object(py)
    });
}

#[pyfunction]
fn minimum_categories_for_coverage(mut cat_to_doc: HashMap<String, Vec<String>>, missed_docs: HashSet<String>, missed_cats: Vec<String>) -> Py<PyAny> {
    // Filter out all documents that are not in the missed documents
    // set.
    let filtered_cat_to_doc: HashMap<String, Vec<String>> = filter_category_map_documents(
        &mut cat_to_doc, &missed_docs, missed_cats.clone()
    );

    // Initialize variables for the search.
    let mut solution: HashSet<String> = HashSet::new();
    let mut visited: Vec<HashSet<String>> = Vec::new();
    let coverage: usize = 0;
    let full_coverage: usize = missed_docs.len();
    let initial_state: (Vec<String>, usize, HashSet<String>) = (missed_cats.clone(), coverage, solution.clone());
    let mut queue: Vec<(Vec<String>, usize, HashSet<String>)> = [initial_state].to_vec();
    let mut is_solved: bool = false;

	// Iterate through a heavily modified BFS to find the smallest
	// combination of categories that would cover the remaining missed
	// documents from the dump.
    while !queue.is_empty() && !is_solved {
        // Pop the state from the queue and unpack it.
        let (mut available_categories, document_coverage, current_solution) = queue.remove(0);
    
        // Check for document coverage. If we have 100% coverage, this
        // is a sign that we have reached a solution state.
        if document_coverage == full_coverage {
            is_solved = true;
            solution = current_solution.clone();
            continue;
        }

        // Skip solutions (category combinations) that have been 
        // visited. Convert the current solution to a set because order    
        if visited.contains(&current_solution) {
            continue;
        }

        // Sort the list of available categories, giving preference to
        // the ones that have more document coverage.
        available_categories.sort_by(|a: &String, b: &String| {
            filtered_cat_to_doc.get(b)
                .unwrap()
                .len()
                .cmp(
                    &filtered_cat_to_doc
                        .get(a)
                        .unwrap()
                        .len()
                )
        });

        let mut best_document_coverage: usize = document_coverage.clone();

        // Initialize vector to keep track of the states generated 
        // below. Also initialize a progress bar for progress tracking.
        let mut options: Vec<(Vec<String>, usize, HashSet<String>)> = Vec::new();
        let pb: ProgressBar = ProgressBar::new(available_categories.len().try_into().unwrap());
        
        // Iterate through each available category in the sorted list.
        // Generate new possible states and append them to the queue.
        for category in &available_categories {
            // Initialize a new (hypothesis) solution by appending the
            // current category to the end of the current solution.
            let mut new_solution: HashSet<String> = current_solution.clone();
            new_solution.insert(category.clone());

            // Compute the new solution's document coverage.
            let mut covered_documents: HashSet<String> = HashSet::new();
            for solution_category in &new_solution {
                if let Some(docs) = filtered_cat_to_doc.get(solution_category) {
                    covered_documents.extend(docs.iter().cloned());
                }
            }
            let new_document_coverage: usize = covered_documents.len();

            // Skip appending states that do not increase the coverage.
            if new_document_coverage <= document_coverage {
                pb.inc(1);
                continue;
            }

            if new_document_coverage > best_document_coverage {
                best_document_coverage = new_document_coverage.clone();
            }
            else {
                pb.inc(1);
                continue;
            }

            // Remove the current category from the list of available 
	        // categories.
            let mut remaining_categories: Vec<String> = available_categories.clone();
            remaining_categories.retain(|remaining_category: &String| remaining_category != category);

            // Create new state tuple and update the options list 
            // accordingly. Also update the list of visited solutions 
            // too.
            let new_state: (Vec<String>, usize, HashSet<String>) = (remaining_categories, new_document_coverage, new_solution.clone());
            options.push(new_state);
            visited.push(new_solution.clone());
            pb.inc(1);
        }

        // Sort list of new state options by highest coverage (priority 
        // goes to solutions that offer higher document coverage). 
        // Append the sorted list to the queue.
        options.sort_by(
            |a: &(Vec<String>, usize, HashSet<String>), b: &(Vec<String>, usize, HashSet<String>)| b.1.cmp(&a.1)
        );
        queue.extend(options);
    }

    // Return the hashset of categories that will guarantee total
    // coverage of the remaining missing documents at with the least
    // categories from the missed categories list.
    return Python::with_gil(|py: Python| {
        solution.clone().to_object(py)
    });
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_search_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_category_map, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_categories_for_coverage, m)?)?;
    Ok(())
}
