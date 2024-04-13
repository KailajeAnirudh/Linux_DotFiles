function setToStorage(name, value, callback) {
    browser.storage.local.set({[name]: value}, function (obj) {
        
        if (typeof callback === 'function') {
            callback();
        }
        
        updateSettingsStringInCloud();
        
    })
}

function setToStorageWithoutSync(name, value, callback) {
    browser.storage.local.set({[name]: value}, callback);
}

function showScreen(name) {
    
    const allAppScreens = document.querySelectorAll(".appScreen");
    
    for (const screen of allAppScreens) {
        screen.removeAttribute("active");
        if (screen.getAttribute("id") == name) {
            screen.setAttribute("active", "");
        }
    }
}

function showDropdown(element) {
    var event;
    event = document.createEvent('MouseEvents');
    event.initMouseEvent('mousedown', true, true, window);
    element.dispatchEvent(event);
};

function getTabs() {
    return browser.tabs.query({})
}

function getCurrentTab() {
    return browser.tabs.query({ currentWindow: true, active: true })
}

function sendCommand(id) {
    getTabs().then((tabs) => {
        for (const tabIndex in tabs) {
            browser.tabs.sendMessage(tabs[tabIndex].id, {
                id: id
            })
        }
    })
}

function sendSelectCommand(id, value) {
    getTabs().then((tabs) => {
        for (const tabIndex in tabs) {
            browser.tabs.sendMessage(tabs[tabIndex].id, {
                type: "select",
                id: id,
                value: value,
            })
        }
    })
}

function getFeaturesArrayOfObjectIds() {
    const obj = getConst;
    let result = [];
      for (const key in obj) {
        if (Array.isArray(obj[key])) {
          // If the value is an array, add its elements to the result
          result.push(...obj[key]);
        } else {
          // If it's not an array, just add the value
          result.push(obj[key]);
        }
      }
      return result;
}

function getSupportedArrayOfObjectIds() {
    const obj = getConstNotSyncing;
    let result = [];
      for (const key in obj) {
        if (Array.isArray(obj[key])) {
          // If the value is an array, add its elements to the result
          result.push(...obj[key]);
        } else {
          // If it's not an array, just add the value
          result.push(obj[key]);
        }
      }
      return result;
}

// Set Route Buttons

const allRouteButtons = document.querySelectorAll(".routerButton");

for (const routeButton of allRouteButtons) {
    
    routeButton.addEventListener("click", function() {
        
        const destination = routeButton.getAttribute("routeto");
        
        // Remove all active states
        
        const allAppScreens = document.querySelectorAll(".appScreen");
        for (const screen of allAppScreens) {
            screen.removeAttribute("active");
        }
        
        // Set active
        
        queryById(destination).setAttribute("active", "");
    });
}