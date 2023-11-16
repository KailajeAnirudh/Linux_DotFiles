/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
var __webpack_exports__ = {};

const originalXhrSend = XMLHttpRequest.prototype.send;
// Inject an empty element into the page to store the last canonicalId and featureReferenceId
// We need this since we don't start listening immediately, so we need the first ID sometimes.
const emptyElement = document.createElement("div");
emptyElement.id = "teleparty-appletv-id-container";
document.body.appendChild(emptyElement);
// Override the send method to capture the request data
XMLHttpRequest.prototype.send = function (data) {
    var _a, _b, _c, _d;
    try {
        const dataJson = JSON.parse(data);
        const newCanonicalId = (_b = (_a = dataJson === null || dataJson === void 0 ? void 0 : dataJson.events) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.featureCanonicalId;
        const newFeatureReferenceId = (_d = (_c = dataJson === null || dataJson === void 0 ? void 0 : dataJson.events) === null || _c === void 0 ? void 0 : _c[0]) === null || _d === void 0 ? void 0 : _d.featureReferenceId;
        if (newCanonicalId && newFeatureReferenceId) {
            emptyElement.setAttribute("data-canonical-id", newCanonicalId);
            emptyElement.setAttribute("data-feature-reference-id", newFeatureReferenceId);
            console.log("Detected new canonicalId and featureReferenceId", newCanonicalId, newFeatureReferenceId);
            window.postMessage({
                type: "idDetected",
                canonicalId: newCanonicalId,
                featureReferenceId: newFeatureReferenceId
            }, "*");
        }
    }
    catch (e) {
        // Do nothing
    }
    return originalXhrSend.apply(this, arguments);
};

/******/ })()
;