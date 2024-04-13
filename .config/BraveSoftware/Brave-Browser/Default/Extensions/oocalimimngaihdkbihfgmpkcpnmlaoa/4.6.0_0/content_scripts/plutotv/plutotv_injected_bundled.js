/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ 8619:
/***/ (function() {


var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const originalFetch = window.fetch;
function createAdBreakLocationsContainerIfNeeded() {
    if (!document.getElementById("tp-ad-break-locations-container")) {
        const adLocContainer = document.createElement("div");
        adLocContainer.id = "tp-ad-break-locations-container";
        document.body.appendChild(adLocContainer);
    }
}
function getAdBreaksAsJson() {
    const adLocContainer = document.getElementById("tp-ad-break-locations-container");
    const adBreakLocations = adLocContainer.getAttribute("data-ad-break-locations");
    return JSON.parse(adBreakLocations);
}
window.fetch = function (input, init) {
    return __awaiter(this, arguments, void 0, function* () {
        const response = yield originalFetch.apply(this, arguments);
        try {
            const data = yield response.clone().json(); // Clone the response to avoid consuming it
            const adBreakLocations = data === null || data === void 0 ? void 0 : data.adBreaks;
            if (adBreakLocations) {
                createAdBreakLocationsContainerIfNeeded();
                const adLocContainer = document.getElementById("tp-ad-break-locations-container");
                adLocContainer.setAttribute("data-ad-break-locations", JSON.stringify(adBreakLocations));
                window.postMessage({ type: "adBreakLocations", adBreakLocations }, "*");
            }
        }
        catch (e) {
            // Do nothing
        }
        return response;
    });
};


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module is referenced by other modules so it can't be inlined
/******/ 	var __webpack_exports__ = {};
/******/ 	__webpack_modules__[8619]();
/******/ 	
/******/ })()
;