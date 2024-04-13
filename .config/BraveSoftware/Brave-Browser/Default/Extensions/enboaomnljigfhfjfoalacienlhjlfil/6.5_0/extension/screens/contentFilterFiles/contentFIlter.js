(function() {

    var activeTabId = 0;
    
    // MARK: - Helpers
    
    function removeItemOnce(arr, value) {
      var index = arr.indexOf(value);
      
      if (index > -1) {
        arr.splice(index, 1);
      }
      return arr;
    }
    
    // MARK: - Functions
    
    // Remove leading symbols in string
    
    function removeLeadingAtSymbols(inputString) {
        return inputString.replace(/^@+/, '');
    }
    
    // Clear field state
        
    function clearFieldState() {
        queryById("contentFilterField").classList.remove("error");
        queryById("contentFilterField").value = "";
    }
    
    // Remove active state from all tabs
    
    function makeUnactiveAllTabs() {
        const filterTabs = document.querySelectorAll("#contentFilterScreen .segmentedPicker .option");

        for (const tab of filterTabs) {
            tab.removeAttribute("active");
        }
    }
    
    // Set label and placeholder

    function setLabelAndPlaceholder() {
        const activeTab = document.querySelector("#contentFilterScreen .segmentedPicker .option[data-id='" + activeTabId + "']");

        document.getElementById("contentFilterField").setAttribute("placeholder", activeTab.getAttribute("data-input"));
    }
    
    // Set actions for delete buttons
    
    function setDeleteButtonsActions() {
        const deleteButtons = querySelectorAll("#contentFilterScreen .filterRuleItem .delete");
        
        for (const index in deleteButtons) {
            const button = deleteButtons[index];
            
            button.onclick = function() {
                
                // Get rule label
                const ruleLabel = button.parentElement.querySelector(".label").innerText;
                
                browser.storage.local.get(getCurrentTabStorageName(), function (obj) {
                    var data = obj[getCurrentTabStorageName()];
                    
                    // Delete from temp array
                    data = removeItemOnce(data, ruleLabel);
                    
                    // Update in storage
                    setToStorage(getCurrentTabStorageName(), data, function() {
                        
                        // Upadte HTML
                        presentTabRules();
                        
                        // Update counter on more screen
                        getFiltersStatus();
                    });
                })
            }
        }
    }
    
    // Get storage constant by current tab
    
    function getCurrentTabStorageName() {
        if (activeTabId == 0) {
            return getConst.filterChannelsRulesData;
        } else if (activeTabId == 1) {
            return getConst.filterVideosRulesData;
        } else if (activeTabId == 2) {
            return getConst.filterCommentsRulesData;
        } else if (activeTabId == 3) {
            return getConst.filterPostsRulesData;
        }
    }
    
    // Present list of rules
    
    function presentRulesListInUI(rules) {
        
        querySelector("#contentFilterScreen .filterRulesWrapper").innerHTML = "";
        
        rules.forEach((item) => {
            const filterRuleItem = document.createElement("div");
            filterRuleItem.classList.add("filterRuleItem");
            const label = document.createElement("div");
            label.classList.add("label");
            label.innerHTML = item;
            const del = document.createElement("div");
            del.classList.add("delete");
            filterRuleItem.appendChild(label);
            filterRuleItem.appendChild(del);
            querySelector("#contentFilterScreen .filterRulesWrapper").appendChild(filterRuleItem);
        })
        
        setDeleteButtonsActions();
    }
    
    // Present rules for specific type
    
    function presentTabRules() {
        
        browser.storage.local.get(getCurrentTabStorageName(), function (obj) {
            const data = obj[getCurrentTabStorageName()] ?? [];
            
            
            if (data.length == 0) {
                querySelector("#contentFilterScreen .filterRulesEmptyMessage").style.display = "block";
                querySelector("#contentFilterScreen .filterRulesWrapper").style.display = "none";
            } else {
                presentRulesListInUI(data);
                
                querySelector("#contentFilterScreen .filterRulesEmptyMessage").style.display = "none";
                querySelector("#contentFilterScreen .filterRulesWrapper").style.display = "block";
            }
        })
    }
    
    // MARK: - Life Cycle
    
    setLabelAndPlaceholder();
    presentTabRules();
    
    // MARK: - Actions
    
    // Tabs click: Channels, Videos, Comments, Posts
    
    const filterTabs = document.querySelectorAll("#contentFilterScreen .segmentedPicker .option");

    for (const tab of filterTabs) {
        tab.onclick = function() {
            makeUnactiveAllTabs();
            clearFieldState();
            this.setAttribute("active", "");
            activeTabId = this.getAttribute("data-id");
            document.getElementById("contentFilterField").setAttribute("placeholder", this.getAttribute("data-input"));
            presentTabRules();
        }
    }
    
    // Add Rule
    
    queryById("contentFilterFieldAddButton").onclick = function() {
        var textFieldValue = queryById("contentFilterField").value;
        
        if (textFieldValue.replaceAll(" ", "").length == 0) {
            queryById("contentFilterField").classList.add("error");
        } else {
            
            var error = false;
            
            // Try to extract link
            const extractionFunctions = {
                0: extractChannelId,
                1: extractVideoId,
                2: extractCommentID,
                3: extractPostID
            };

            if (textFieldValue.includes("youtube.com/") && extractionFunctions.hasOwnProperty(activeTabId)) {
                const extracted = extractionFunctions[activeTabId](textFieldValue);
                if (extracted) {
                    textFieldValue = extracted;
                } else {
                    queryById("contentFilterField").classList.add("error");
                    error = true;
                }
            }
            
            if (activeTabId == 0) {
                if (textFieldValue.substring(0, 1) == "@") {
                    textFieldValue = removeLeadingAtSymbols(textFieldValue);
                }
            }
            
            if (!error) {
                const currentTab = getCurrentTabStorageName();
                
                browser.storage.local.get(currentTab, function (obj) {
                    var data = obj[currentTab] ?? [];
                    
                    if (!data.includes(textFieldValue)) {
                        
                        data.unshift(textFieldValue);
                        
                        // Add to storage
                        setToStorage(currentTab, data, function() {
                            
                            // Update HTML
                            presentTabRules();
                            
                            // Update counter on more screen
                            getFiltersStatus();
                        });
                    }
                })
            }

        }
        
        queryById("contentFilterField").value = "";
    }
    
    // Tap on toggle
    
    queryById("blocklistFilterContextButtons").onclick = function() {
        const button = queryById("blocklistFilterContextButtons").checked;
        
        setToStorage(getConst.blocklistContextMenuButtonsData, button, function() {
            
        });
    }
    
})();

