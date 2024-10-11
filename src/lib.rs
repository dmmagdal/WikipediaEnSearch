use pyo3::prelude::*;
// use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::collections::HashSet;
// use std::process::exit;
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
fn verify_filtered_category_map(
    cat_to_doc: HashMap<String, Vec<String>>,
    missed_docs: HashSet<String>,
    missed_cats: Vec<String>
) {
    // Initialize a progress bar.
    let pb: ProgressBar = ProgressBar::new(missed_cats.len().try_into().unwrap());

    // Iterate through each category in the missed categories vector.
    for category in missed_cats {
        // If the category is a valid key within the category to 
        // document hashmap, verify the documents.
        if let Some(docs) = cat_to_doc.get(&category) {
            // Verify documents by
            // 1) convert the documents (value) for that category (key)
            //  in the hashset. 
            // 2) take the intersection of the documents (value) and
            //  the missed documents hashset.
            // 3) compare the length of the intersection with the 
            //  length of the documents hashset. If the lengths are the
            //  same, then the documents for that category are 
            //  exclusively a subset of the missed documents hashset
            //  and the documents for that category are valid.
            let doc_hashset: HashSet<String> = docs.clone().into_iter().collect();
            let intersection: HashSet<_> = doc_hashset.intersection(&missed_docs).collect();
            assert_eq!(doc_hashset.len(), intersection.len());
        }

        pb.inc(1);
    }
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
fn minimum_categories_for_coverage(
    mut cat_to_doc: HashMap<String, Vec<String>>, 
    missed_docs: HashSet<String>, 
    missed_cats: Vec<String>, 
    use_bfs: bool
) -> Py<PyAny> {
    // Filter out all documents that are not in the missed documents
    // set.
    let filtered_cat_to_doc: HashMap<String, Vec<String>> = filter_category_map_documents(
        &mut cat_to_doc, &missed_docs, missed_cats.clone()
    );

    // Verify filter worked.
    verify_filtered_category_map(
        filtered_cat_to_doc.clone(),
        missed_docs.clone(),
        missed_cats.clone()
    );

    // Remove missed categories that have 0 document coverage.
    let mut filtered_missed_cats: HashSet<String> = HashSet::from_iter(missed_cats.clone());
    for category in &missed_cats {
        if let Some(docs) = filtered_cat_to_doc.get(category) {
            if docs.len() == 0 {
                filtered_missed_cats.remove(category);
            }
        }
    }
    let missed_cats_vec: Vec<String> = Vec::from_iter(filtered_missed_cats);

    // Initialize variables for the search.
    let mut solution: HashSet<String> = HashSet::new();
    let mut visited: Vec<HashSet<String>> = Vec::new();
    let coverage: usize = 0;
    let full_coverage: usize = missed_docs.len();
    // let initial_state: (Vec<String>, usize, HashSet<String>) = (missed_cats.clone(), coverage, solution.clone());
    let initial_state: (Vec<String>, usize, HashSet<String>) = (missed_cats_vec.clone(), coverage, solution.clone());
    let mut queue: Vec<(Vec<String>, usize, HashSet<String>)> = [initial_state].to_vec();
    let mut is_solved: bool = false;

    if use_bfs {
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
    
            // Initialize vector to keep track of the states generated 
            // below. Also initialize a progress bar for progress tracking.
            let mut options: Vec<(Vec<String>, usize, HashSet<String>)> = Vec::new();
            let pb: ProgressBar = ProgressBar::new(available_categories.len().try_into().unwrap());
            // let mut best_coverage: usize = 0;
    
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
    
                // if new_document_coverage > best_coverage {
                //     best_coverage = new_document_coverage;
                // }
                // else {
                //     pb.inc(1);
                //     continue;
                // }
    
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
    }
    else {
        // Iterate through a heavily modified BFS to find the smallest
        // combination of categories that would cover the remaining missed
        // documents from the dump.
        while !queue.is_empty() && !is_solved {
            // Pop the state from the queue and unpack it.
            let (mut available_categories, document_coverage, current_solution) = queue.pop().unwrap();
        
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
    
            // Initialize vector to keep track of the states generated 
            // below. Also initialize a progress bar for progress tracking.
            let mut options: Vec<(Vec<String>, usize, HashSet<String>)> = Vec::new();
            let pb: ProgressBar = ProgressBar::new(available_categories.len().try_into().unwrap());
            // let mut best_coverage: usize = 0;
    
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
    
                // if new_document_coverage > best_coverage {
                //     best_coverage = new_document_coverage;
                // }
                // else {
                //     pb.inc(1);
                //     continue;
                // }
    
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
    }

    assert_eq!(is_solved, true);

    // Return the hashset of categories that will guarantee total
    // coverage of the remaining missing documents at with the least
    // categories from the missed categories list.
    return Python::with_gil(|py: Python| {
        solution.clone().to_object(py)
    });
}

#[pyfunction]
fn minimum_categories_for_coverage_new(
    mut cat_to_doc: HashMap<String, Vec<String>>, 
    missed_docs: HashSet<String>, 
    missed_cats: Vec<String>, 
    chunk_size: usize
) -> Py<PyAny> {
    // Filter out all documents that are not in the missed documents
    // set.
    let filtered_cat_to_doc: HashMap<String, Vec<String>> = filter_category_map_documents(
        &mut cat_to_doc, &missed_docs, missed_cats.clone()
    );

    // Verify filter worked.
    verify_filtered_category_map(
        filtered_cat_to_doc.clone(),
        missed_docs.clone(),
        missed_cats.clone()
    );

    // Remove missed categories that have 0 document coverage.
    let mut filtered_missed_cats: HashSet<String> = HashSet::from_iter(missed_cats.clone());
    for category in &missed_cats {
        if let Some(docs) = filtered_cat_to_doc.get(category) {
            if docs.len() == 0 {
                filtered_missed_cats.remove(category);
            }
        }
    }
    let mut missed_cats_vec: Vec<String> = Vec::from_iter(filtered_missed_cats);

    // Sort missed categories by the number of (missed) documents
    // they correspond to.
    missed_cats_vec.sort_by(|a: &String, b: &String| {
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

    // Chunk the missed categories.
    // let chunk_size: usize = 1_000;
    let mut category_chunks: Vec<Vec<String>> = Vec::new();
    let mut current_chunk: Vec<String> = Vec::new();
    let mut current_len: usize = 0;

    for category in missed_cats_vec {
        if current_len + category.len() > chunk_size {
            category_chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_len = 0;
        }
        current_len += category.len();
        current_chunk.push(category);
    }

    if !current_chunk.is_empty() {
        category_chunks.push(current_chunk);
    }

    let mut solution: HashSet<String> = HashSet::new();
    let mut is_solved: bool = false;
    let mut coverage: usize = 0;
    let full_coverage: usize = missed_docs.len();
    for (idx, chunk) in category_chunks.iter().enumerate() {
        println!("Searching chunk {}/{}", idx + 1, category_chunks.len());
        println!("Best coverage {}/{}", coverage, full_coverage);

        // Local variables for state search for this chunk.
        let initial_state: (Vec<String>, usize, HashSet<String>) = (chunk.clone(), coverage, solution.clone());
        let mut queue: Vec<(Vec<String>, usize, HashSet<String>)> = [initial_state].to_vec();
        let mut visited: Vec<HashSet<String>> = Vec::new();

        // Local variables to track coverage and best solution.
        let mut local_solution: HashSet<String> = solution.clone();
        let mut local_coverage: usize = coverage.clone();

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
                coverage = document_coverage.clone();

                println!("New coverage {}/{}", coverage, full_coverage);

                continue;
            }

            // Skip solutions (category combinations) that have been 
            // visited. Convert the current solution to a set because
            // order of categories in solution states does not matter.
            if visited.contains(&current_solution) {
                continue;
            }
            else {
                visited.push(current_solution.clone());
            }

            // Set solution if its coverage beats the current coverage.
            println!("Document coverage v local coverage {} {}", document_coverage, local_coverage);
            // if document_coverage > coverage {
            if document_coverage > local_coverage {
                local_solution = current_solution.clone();
                local_coverage = document_coverage.clone();

                // println!("New coverage {}/{}", coverage, full_coverage);
                println!("New coverage {}/{}", local_coverage, full_coverage);
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

                println!("New Document Coverage {}", new_document_coverage);
                println!("Document Coverage {}", document_coverage);

                // Only append new potential states that offer more 
                // document coverage than the current one.
                if new_document_coverage > document_coverage {
                    println!("New coverage > document coverage");
                    
                    // Filter out the category from the list of
                    // remaining categories.
                    let mut remaining_categories: Vec<String> = available_categories.clone();
                    remaining_categories.retain(|remaining_category| remaining_category != category);
    
                    // Initialize the new state and pass it into the
                    // queue.
                    let new_state: (Vec<String>, usize, HashSet<String>) = (remaining_categories.clone(), new_document_coverage.clone(), new_solution.clone());
                    options.push(new_state);

                }

                // Increment the progress bar
                pb.inc(1);
            }

            // Sort list of new state options by highest coverage 
            // (priority goes to solutions that offer higher document
            // coverage). Append the sorted list to the queue.
            options.sort_by(
                |a: &(Vec<String>, usize, HashSet<String>), b: &(Vec<String>, usize, HashSet<String>)| b.1.cmp(&a.1)
            );
            queue.extend(options);
        }

        println!("Local coverage {}", local_coverage);
        println!("Global coverage {}", coverage);

        // Update global solution and coverage after processing this chunk
        if local_coverage > coverage {
            solution = local_solution;
            coverage = local_coverage;

            println!("Updated global solution with chunk solution. Coverage: {}/{}", coverage, full_coverage);
        }

        if is_solved {
            break;
        }
    }

    assert_eq!(is_solved, true);

    // Return the hashset of categories that will guarantee total
    // coverage of the remaining missing documents at with the least
    // categories from the missed categories list.
    return Python::with_gil(|py: Python| {
        solution.clone().to_object(py)
    });
}


#[pyfunction]
fn get_leaves(graph: HashMap<String, Vec<String>>, max_depth: u32, number_of_categories: u64) -> Py<PyAny> {
    // Single pass BFS (get depth while traversing and apply filter).
    let mut visited: HashSet<String> = HashSet::new();
    let start_node: (String, u32) = ("Main topic classifications".to_string(), 0);
    let mut queue: Vec<(String, u32)> = [start_node].to_vec();
    let mut valid_categories: Vec<(String, u32)> = Vec::new();

    let pb: ProgressBar = ProgressBar::new(number_of_categories);

    while queue.len() != 0 {
        let (category, depth) = queue.remove(0);

        if visited.contains(&category) {
            continue;
        }

        visited.insert(category.clone());

        // if depth <= max_depth && !graph.contains_key(&category) {
        //     valid_categories.push(category.clone());
        // }

        if let Some(children) = graph.get(&category) {
            if children.len() == 0 && depth <= max_depth {
                valid_categories.push((category.clone(), depth ));
            }
            else {
                for child in children {
                    queue.push((child.to_string(), depth + 1));
                }
            }
        }

        pb.inc(1);
    }

    return Python::with_gil(|py: Python| {
        valid_categories.clone().to_object(py)
    });
}


/// A Python module implemented in Rust.
#[pymodule]
fn rust_search_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_category_map, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_categories_for_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(verify_filtered_category_map, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_categories_for_coverage_new, m)?)?;
    m.add_function(wrap_pyfunction!(get_leaves, m)?)?;
    Ok(())
}
