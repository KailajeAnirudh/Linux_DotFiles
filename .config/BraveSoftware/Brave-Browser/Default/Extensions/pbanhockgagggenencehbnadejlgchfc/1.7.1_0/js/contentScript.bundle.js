(()=>{var e={2022:(e,t,n)=>{var r=n(6540),a=n(1912),i=n(9225),o=n(3427),s=n(184),l=n(2831),c=n(1390),d=n(1614),u=n(2838),p=n(9371);function h(e){return e&&e.__esModule?e:{default:e}}function f(e){if(e&&e.__esModule)return e;var t=Object.create(null);return e&&Object.keys(e).forEach((function(n){if("default"!==n){var r=Object.getOwnPropertyDescriptor(e,n);Object.defineProperty(t,n,r.get?r:{enumerable:!0,get:function(){return e[n]}})}})),t.default=e,Object.freeze(t)}var m=f(r),g=h(a),y=h(i),v=h(o),b=h(s),x=h(u),w=Object.defineProperty,_=Object.defineProperties,k=Object.getOwnPropertyDescriptors,S=Object.getOwnPropertySymbols,E=Object.prototype.hasOwnProperty,A=Object.prototype.propertyIsEnumerable,P=(e,t,n)=>t in e?w(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,j=(e,t)=>{for(var n in t||(t={}))E.call(t,n)&&P(e,n,t[n]);if(S)for(var n of S(t))A.call(t,n)&&P(e,n,t[n]);return e},N=(e,t)=>_(e,k(t)),C=e=>"symbol"==typeof e?e:e+"",O=(e,t)=>{var n={};for(var r in e)E.call(e,r)&&t.indexOf(r)<0&&(n[r]=e[r]);if(null!=e&&S)for(var r of S(e))t.indexOf(r)<0&&A.call(e,r)&&(n[r]=e[r]);return n},T=(e,t,n)=>new Promise(((r,a)=>{var i=e=>{try{s(n.next(e))}catch(e){a(e)}},o=e=>{try{s(n.throw(e))}catch(e){a(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(i,o);s((n=n.apply(e,t)).next())})),I="1.38.33",M="NOT_STARTED_STEP",D="COMPLETED_FLOW",L="SKIPPED_FLOW",$="STARTED_FLOW",R="NOT_STARTED_FLOW",F="COMPLETED_STEP",U="STARTED_STEP";function B(){let{publicApiKey:e,userId:t,apiUrl:n}=m.default.useContext(Yn);return{config:r.useMemo((()=>({headers:{Authorization:`Bearer ${e}`,"Content-Type":"application/json","X-Frigade-SDK-Version":I,"X-Frigade-SDK-Platform":"React"}})),[e,t]),apiUrl:r.useMemo((()=>`${n}/v1/public/`),[n])}}var z="frigade-last-call-at-",V="frigade-last-call-data-";function H(){let{shouldGracefullyDegrade:e,readonly:t}=m.default.useContext(Yn);return(n,r)=>T(this,null,(function*(){if(t&&("POST"===r.method||"PUT"===r.method||"DELETE"===r.method))return Y();if(e)return Y();let a,i=z+n,o=V+n;if("undefined"!=typeof window&&window.localStorage&&r&&r.body&&"POST"===r.method){let e=window.localStorage.getItem(i),t=window.localStorage.getItem(o);if(e&&t&&t==r.body){let t=new Date(e);if((new Date).getTime()-t.getTime()<1e3)return Y()}"undefined"!=typeof window&&(window.localStorage.setItem(i,(new Date).toISOString()),window.localStorage.setItem(o,r.body))}try{a=yield fetch(n,r)}catch(e){return Y(e)}return a?a.ok?a:Y(a.statusText):Y()}))}function Y(e){return{json:()=>({})}}function W(){let{publicApiKey:e,shouldGracefullyDegrade:t}=m.default.useContext(Yn);return{verifySDKInitiated:function(){return!t&&!!e}}}function q(e,t,n,r={}){return fetch(e,r).catch((function(a){let i=n-1;if(!i)throw a;return function(e){return new Promise((t=>setTimeout(t,e)))}(t).then((()=>q(e,t,i,r)))}))}function K(){let{openFlowStates:e,setOpenFlowStates:t,hasActiveFullPageFlow:n,setCompletedFlowsToKeepOpenDuringSession:a,completedFlowsToKeepOpenDuringSession:i}=r.useContext(Yn);return{getOpenFlowState:function(t,n=!1){var r;return null!=(r=e[t])?r:n},setOpenFlowState:function(e,n){t((t=>N(j({},t),{[e]:n})))},resetOpenFlowState:function(e){t((t=>{let n=O(t,[C(e)]);return j({},n)}))},hasOpenModals:function(t){return Object.entries(e).some((([e,n])=>n&&e!=t))||n},setKeepCompletedFlowOpenDuringSession:function(e){i.includes(e)||a((t=>[...t,e]))},shouldKeepCompletedFlowOpenDuringSession:function(e){return i.includes(e)}}}function G(e){return"object"==typeof e&&null!==e&&!Array.isArray(e)}function Z(...e){let t=e.shift(),n=1===e.length?e[0]:Z(...e);if(!G(t)||!G(n))throw new Error("deepmerge can only merge Objects");let r=b.default(t);return Object.entries(n).forEach((([e,t])=>{G(t)?void 0!==r[e]?Object.assign(r,{[e]:Z(r[e],b.default(t))}):Object.assign(r,{[e]:b.default(t)}):Array.isArray(t)?void 0!==r[e]?Object.assign(r,{[e]:[...r[e],...b.default(t)]}):Object.assign(r,{[e]:b.default(t)}):Object.assign(r,{[e]:t})})),r}function J(e){try{return JSON.parse(e)}catch(e){return null}}var X="unknown";function Q(){let{config:e,apiUrl:t}=B(),{publicApiKey:n,userId:a,organizationId:i,flows:o,setShouldGracefullyDegrade:s,readonly:l}=r.useContext(Yn),{resetOpenFlowState:c}=K(),[d,u]=r.useState(!1),p={data:o.map((e=>({flowId:e.id,flowState:D,lastStepId:null,userId:a,foreignUserId:a,stepStates:{},shouldTrigger:!1})))},h=t=>q(t,100,2,e).then((e=>{if(e.ok)return e.json();throw new Error("Failed to fetch user flow states")})).catch((e=>(s(!0),p))),f=n&&o&&a?`${t}userFlowStates?foreignUserId=${encodeURIComponent(a)}${i?`&foreignUserGroupId=${encodeURIComponent(i)}`:""}`:null,{data:m,isLoading:g,mutate:b,error:x}=l?v.default(f,h):y.default(f,h,{revalidateOnFocus:!0,revalidateIfStale:!0,keepPreviousData:!0,revalidateOnMount:!1,errorRetryInterval:1e4,errorRetryCount:3,onError:()=>p,onLoadingSlow:()=>p}),w=null==m?void 0:m.data;return r.useEffect((()=>{!d&&!g&&w&&u(!0)}),[w,d,g]),{userFlowStatesData:w,isLoadingUserFlowStateData:!d,mutateUserFlowState:b,optimisticallyMarkFlowCompleted:function(e){return T(this,null,(function*(){if(w&&!l){let t=w.find((t=>t.flowId===e));t&&t.flowState!==D&&(t.flowState=D),yield b(Promise.resolve(Z(m,{data:w})),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1})}}))},optimisticallyMarkFlowSkipped:function(e){return T(this,null,(function*(){if(w&&!l){let t=w.find((t=>t.flowId===e));t&&t.flowState!==L&&(t.flowState=L),yield b(Promise.resolve(Z(m,{data:w})),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1})}}))},optimisticallyMarkFlowNotStarted:function(e){return T(this,null,(function*(){if(w){let t=w.find((t=>t.flowId===e));t&&t.flowState!==R&&(t.flowState=R,t.lastStepId=X,Object.keys(t.stepStates).forEach((e=>{t.stepStates[e].actionType=M,t.stepStates[e].createdAt=(new Date).toISOString()})),yield b(Z(m,{data:w}),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1}),c(e))}}))},optimisticallyMarkStepCompleted:function(e,t,n){return T(this,null,(function*(){var r,a,i;if(w){let s=w.find((t=>t.flowId===e));if(s){let l=o.find((t=>t.slug===e)),c=J(null==l?void 0:l.data),d=null!=(a=null!=(r=null==c?void 0:c.steps)?r:null==c?void 0:c.data)?a:[],u=d.findIndex((e=>e.id===t)),p=d&&d.length>u+1?d[u+1]:null;p&&(null!=(i=s.stepStates[p.id])&&i.hidden||(s.lastStepId=p.id)),s.stepStates[t]=n,s.flowState=$}l||(yield b(Promise.resolve(Z(m,{data:w})),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1}))}}))},optimisticallyMarkStepNotStarted:function(e,t,n){return T(this,null,(function*(){if(w){let r=w.find((t=>t.flowId===e));r&&(r.stepStates[t]=n),yield b(Promise.resolve(Z(m,{data:w})),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1})}}))},optimisticallyMarkStepStarted:function(e,t,n){return T(this,null,(function*(){if(w){let r=w.find((t=>t.flowId===e));r&&(r.lastStepId=t,r.stepStates[t]=n,r.flowState=$),l||(yield b(Z(m,{data:w}),{optimisticData:Z(m,{data:w}),revalidate:!1,rollbackOnError:!1}))}}))},error:x}}function ee(){let{config:e,apiUrl:t}=B(),{userFlowStatesData:n,mutateUserFlowState:a}=Q(),{failedFlowResponses:i,setFailedFlowResponses:o,flowResponses:s,setFlowResponses:l}=r.useContext(Yn),[c,d]=r.useState(new Set),[u,p]=r.useState(new Set),h=H();function f(n){let r=JSON.stringify(n);if(c.has(r))return null;c.add(r),d(c),u.add(n),p(u);let a=null==s?void 0:s.find((e=>e.flowSlug===n.flowSlug&&e.stepId===n.stepId&&e.actionType===n.actionType&&e.createdAt===n.createdAt));return h(`${t}flowResponses`,N(j({},e),{method:"POST",body:r})).then((e=>{200!==e.status&&201!==e.status?o([...i,n]):a||l((e=>[...null!=e?e:[],n]))}))}return{addResponse:function(e){return T(this,null,(function*(){e.foreignUserId&&(e.actionType===$||e.actionType===R||e.actionType===D||e.actionType===U||e.actionType===F||e.actionType===L||e.actionType===M)&&(yield f(e))}))},setFlowResponses:l,getFlowResponses:function(){let e=[];return null==n||n.forEach((t=>{if(t&&t.stepStates&&0!==Object.keys(t.stepStates).length)for(let n in t.stepStates){let r=t.stepStates[n];e.push({foreignUserId:t.foreignUserId,flowSlug:t.flowId,stepId:r.stepId,actionType:r.actionType,data:{},createdAt:new Date(r.createdAt),blocked:r.blocked,hidden:r.hidden})}})),[...e,...s]}}}var te,ne=/user.flow\(([^\)]+)\) == '?COMPLETED_FLOW'?/gm,re=e=>{let t=ne.exec(e);if(null===t)return null;let n=null;return t.forEach(((e,t)=>{let r=ae(e,"'","");r.startsWith("flow_")&&(n=r)})),n},ae=function(e,t,n){return e.replace(new RegExp(t,"g"),n)},ie=((te=ie||{}).CHECKLIST="CHECKLIST",te.FORM="FORM",te.TOUR="TOUR",te.SUPPORT="SUPPORT",te.CUSTOM="CUSTOM",te.BANNER="BANNER",te.EMBEDDED_TIP="EMBEDDED_TIP",te.NPS_SURVEY="NPS_SURVEY",te.ANNOUNCEMENT="ANNOUNCEMENT",te);function oe(){let{config:e,apiUrl:t}=B(),{flows:n,setFlows:a,userId:i,organizationId:o,publicApiKey:s,customVariables:l,setCustomVariables:c,hasActiveFullPageFlow:d,setHasActiveFullPageFlow:u,setFlowResponses:p,setShouldGracefullyDegrade:h,shouldGracefullyDegrade:f,readonly:m,flowDataOverrides:g}=r.useContext(Yn),v={data:[]},{verifySDKInitiated:b}=W(),{addResponse:x,getFlowResponses:w}=ee(),{mutateUserFlowState:_,userFlowStatesData:k,isLoadingUserFlowStateData:S,optimisticallyMarkFlowCompleted:E,optimisticallyMarkFlowSkipped:A,optimisticallyMarkFlowNotStarted:P,optimisticallyMarkStepCompleted:C,optimisticallyMarkStepNotStarted:O,optimisticallyMarkStepStarted:I}=Q(),{data:z,error:V,isLoading:H}=y.default(s?`${t}flows${m?"?readonly=true":""}`:null,(t=>q(t,100,2,e).then((e=>e.ok?e.json():(h(!0),v))).catch((e=>(h(!0),v)))),{keepPreviousData:!0});function Y(e){if(H)return null;let t=n.find((t=>t.slug===e));return!t&&n.length>0&&!S&&!H?null:(t&&g&&g[e]&&(t.data=g[e]),!1!==(null==t?void 0:t.active)||m?t:null)}function K(e){var t,n,r,a,i;if(!Y(e))return[];let o=null==(t=Y(e))?void 0:t.data;return o?(o=G(o),(null!=(i=null!=(a=null==(n=J(o))?void 0:n.data)?a:null==(r=J(o))?void 0:r.steps)?i:[]).map((t=>{let n=he(t);return N(j({handleSecondaryButtonClick:()=>{!0===t.skippable&&ne(e,t.id,{skipped:!0})}},t),{complete:ce(e,t.id)===F||n>=1,started:ce(e,t.id)===U||ce(e,t.id)===F,currentlyActive:null==k?void 0:k.some((n=>n.flowId==e&&n.lastStepId===t.id)),blocked:de(e,t.id),hidden:ue(e,t.id),handlePrimaryButtonClick:()=>{(!t.completionCriteria&&(t.autoMarkCompleted||void 0===t.autoMarkCompleted)||t.completionCriteria&&!0===t.autoMarkCompleted)&&ne(e,t.id)},progress:n})})).filter((e=>!0!==e.hidden))):[]}function G(e){return e.replaceAll(/\${(.*?)}/g,((e,t)=>void 0===l[t]?"":String(l[t]).replace(/[\u00A0-\u9999<>\&]/g,(function(e){return"&#"+e.charCodeAt(0)+";"})).replaceAll(/[\\]/g,"\\\\").replaceAll(/[\"]/g,'\\"').replaceAll(/[\/]/g,"\\/").replaceAll(/[\b]/g,"\\b").replaceAll(/[\f]/g,"\\f").replaceAll(/[\n]/g,"\\n").replaceAll(/[\r]/g,"\\r").replaceAll(/[\t]/g,"\\t")))}function Z(e,t){c((n=>N(j({},n),{[e]:t})))}r.useEffect((()=>{V||z&&z.data&&a(z.data)}),[z,V]);let X=r.useCallback(((e,t,n)=>T(this,null,(function*(){if(!b())return;let r={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:t,actionType:U,data:null!=n?n:{},createdAt:new Date,blocked:!1,hidden:!1};le(r)&&(yield I(e,t,r),x(r))}))),[i,o,k]),te=r.useCallback(((e,t,n)=>T(this,null,(function*(){if(!b())return;let r={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:t,actionType:M,data:null!=n?n:{},createdAt:new Date,blocked:!1,hidden:!1};le(r)&&(yield O(e,t,r),x(r))}))),[i,o,k]),ne=r.useCallback(((e,t,n)=>T(this,null,(function*(){if(!b())return;let r={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:t,actionType:F,data:null!=n?n:{},createdAt:new Date,blocked:!1,hidden:!1};le(r)&&(yield C(e,t,r),x(r))}))),[i,o,k]),ae=r.useCallback(((e,t)=>T(this,null,(function*(){if(!b()||fe(e)===R)return;let n={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:"unknown",actionType:R,data:null!=t?t:{},createdAt:new Date,blocked:!1,hidden:!1};yield P(e),le(n)&&x(n)}))),[i,o,k]),ie=r.useCallback(((e,t)=>T(this,null,(function*(){if(!b())return;let n={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:"unknown",actionType:$,data:null!=t?t:{},createdAt:new Date,blocked:!1,hidden:!1};le(n)&&x(n)}))),[i,o,k]),oe=r.useCallback(((e,t)=>T(this,null,(function*(){if(!b())return;let n={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:"unknown",actionType:D,data:null!=t?t:{},createdAt:new Date,blocked:!1,hidden:!1};le(n)&&(yield E(e),x(n))}))),[i,o,k]),se=r.useCallback(((e,t)=>T(this,null,(function*(){if(!b())return;let n={foreignUserId:i,foreignUserGroupId:null!=o?o:null,flowSlug:e,stepId:"unknown",actionType:L,data:null!=t?t:{},createdAt:new Date,blocked:!1,hidden:!1};le(n)&&(yield A(e),x(n))}))),[i,o,k]);function le(e){var t;if(void 0===k)return!1;if(k){let n=k.find((t=>t.flowId===e.flowSlug));if(e.actionType===M&&(null==n||!n.stepStates[e.stepId]||n.stepStates[e.stepId].actionType===M))return!1;if(n&&(null==(t=n.stepStates[e.stepId])?void 0:t.actionType)===e.actionType){if(e.actionType===F&&(!e.data||JSON.stringify(e.data)===JSON.stringify({})))return!1;let t=Object.keys(n.stepStates).sort(((e,t)=>{let r=new Date(n.stepStates[e].createdAt),a=new Date(n.stepStates[t].createdAt);return r.getTime()-a.getTime()}));if(n.stepStates[t[t.length-1]].actionType===e.actionType&&e.stepId===t[t.length-1])return!1}if(n&&n.flowState===D&&e.actionType===D)return!1}return!0}function ce(e,t){let n=pe(e,t);return S?null:n?n.actionType:M}function de(e,t){let n=pe(e,t);return!!n&&n.blocked}function ue(e,t){let n=pe(e,t);return!!n&&n.hidden}function pe(e,t){var n;if(S)return null;let r=null==k?void 0:k.find((t=>t.flowId===e));return r&&r.stepStates[t]&&null!=(n=r.stepStates[t])?n:null}function he(e){if(!e.completionCriteria)return;let t=re(e.completionCriteria);if(null===t)return;let n=me(t),r=ge(t);return 0===r?void 0:n/r}function fe(e){let t=null==k?void 0:k.find((t=>t.flowId===e));return t?t.flowState:null}function me(e){let t=K(e);return 0===t.length?0:t.filter((t=>ce(e,t.id)===F)).length}function ge(e){return K(e).length}function ye(e){if(m)return!1;if(S||f)return!0;if(null!=e&&e.targetingLogic&&k){let t=k.find((t=>t.flowId===e.slug));if(t)return!1===t.shouldTrigger}return!!(null!=e&&e.targetingLogic&&i&&i.startsWith("guest_"))}return{getAllFlows:function(){return n},getFlow:Y,getFlowData:function(e){let t=n.find((t=>t.slug===e));return t?(g&&g[e]&&(t.data=g[e]),J(t.data)):null},isLoading:S||H,getStepStatus:ce,getFlowSteps:K,getCurrentStepIndex:function(e){var t;let n=function(e){var t,n;if(S||!k)return null;if(fe(e)===R)return null!=(t=K(e)[0])?t:null;let r=null==(n=k.find((t=>t.flowId===e)))?void 0:n.lastStepId;return r?K(e).find((e=>e.id===r)):null}(e);if(!n)return 0;let r=null!=(t=K(e).findIndex((e=>e.id===n.id)))?t:0;return ce(e,n.id)===F&&r<K(e).length-1?r+1:r},markStepStarted:X,markStepCompleted:ne,markFlowNotStarted:ae,markFlowStarted:ie,markFlowCompleted:oe,markFlowSkipped:se,markStepNotStarted:te,getFlowStatus:fe,getNumberOfStepsCompleted:me,getNumberOfSteps:ge,targetingLogicShouldHideFlow:ye,setCustomVariable:Z,updateCustomVariables:function(e){!S&&!H&&e&&JSON.stringify(l)!=JSON.stringify(j(j({},l),e))&&Object.keys(e).forEach((t=>{Z(t,e[t])}))},customVariables:l,getStepOptionalProgress:he,getFlowMetadata:function(e){var t;if(!Y(e))return[];let n=Y(e).data;return n?(n=G(n),null!=(t=JSON.parse(n))?t:{}):[]},isStepBlocked:de,isStepHidden:ue,hasActiveFullPageFlow:d,setHasActiveFullPageFlow:u,isFlowAvailableToUser:function(e){let t=Y(e);return!1!==(null==t?void 0:t.active)&&!ye(Y(e))},refresh:function(){i&&_()},isDegraded:f}}var se="guest_";function le(){let{userId:e,organizationId:t,setUserId:n,setUserProperties:a,shouldGracefullyDegrade:i}=r.useContext(Yn),{config:o,apiUrl:s}=B(),{mutateUserFlowState:l}=Q(),c=H(),{verifySDKInitiated:d}=W();function u(e){return`frigade-user-registered-${e}`}r.useEffect((()=>{if(e&&!t){if(e.startsWith(se))return;let t=u(e);localStorage.getItem(t)||(c(`${s}users`,N(j({},o),{method:"POST",body:JSON.stringify({foreignId:e})})),localStorage.setItem(t,"true"))}}),[e,i,t]);let p=r.useCallback((t=>T(this,null,(function*(){if(!d())return;let n={foreignId:e,properties:t};yield c(`${s}users`,N(j({},o),{method:"POST",body:JSON.stringify(n)})),a((e=>j(j({},e),t))),l()}))),[e,o,i,l]),h=r.useCallback(((t,n)=>T(this,null,(function*(){if(!d())return;let r={foreignId:e,events:[{event:t,properties:n}]};yield c(`${s}users`,N(j({},o),{method:"POST",body:JSON.stringify(r)})),l()}))),[e,o,l]),f=r.useCallback(((e,t)=>T(this,null,(function*(){if(d())if(t&&Object.keys(t).length>0){let r=u(e);localStorage.setItem(r,"true"),n(e);let i={foreignId:e,properties:t};yield c(`${s}users`,N(j({},o),{method:"POST",body:JSON.stringify(i)})),a((e=>j(j({},e),t))),l()}else n(e)}))),[o,i,l]),m=r.useCallback((e=>T(this,null,(function*(){if(!d())return;let t="undefined"!=typeof window?localStorage.getItem(Nn):null;if(!t)return;let n={foreignId:e,linkGuestId:t};yield c(`${s}users`,N(j({},o),{method:"POST",body:JSON.stringify(n)})),l()}))),[o,i,l]);return{userId:e,setUserId:n,setUserIdWithProperties:f,addPropertiesToUser:p,trackEventForUser:h,linkExistingGuestSessionToUser:m}}var ce="fr-",de="cfr-";function ue(e,t){let n=`${ce}${e}`;if(!t)return n;if(t.styleOverrides&&t.styleOverrides[e]){if("string"==typeof t.styleOverrides[e])return n+" "+t.styleOverrides[e];if("object"==typeof t.styleOverrides[e])return n+" "+de+e}return n}function pe(e){if(!e.className||-1!==e.className.indexOf(de))return"";let t=e.className.replace(/\s+/g," ").split(" ");return 1==t.length&&t[0].startsWith(ce)?"":`:not(${t.map((e=>`.${e}`)).join(", ")})`}function he(e){return e.replace(/([a-z0-9]|(?=[A-Z]))([A-Z])/g,"$1-$2").toLowerCase()}function fe(e){return null!=e&&e.styleOverrides?Object.keys(e.styleOverrides).map((t=>`${he(t)}: ${e.styleOverrides[t]};`)).join(" "):""}function me(...e){return e.filter(Boolean).join(" ")}function ge(e){return e.charAt(0).toUpperCase()+e.slice(1)}var ye=g.default.div`
  display: flex;
  justify-content: center;
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  ${e=>pe(e)} {
    background-color: rgba(0, 0, 0, 0.6);
    z-index: 1400;
  }
  animation-duration: 0.15s;
  animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
  animation-name: fadeIn;

  @keyframes fadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
`,ve=({onClose:e,appearance:t})=>m.default.createElement(ye,{className:ue("modalBackground",t),onClick:()=>e()}),be=g.default.div`
  :hover {
    opacity: 0.8;
  }
`,xe=()=>m.default.createElement(be,null,m.default.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",width:"20",height:"20",fill:"none",viewBox:"0 0 20 20"},m.default.createElement("path",{stroke:"currentColor",strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:"1.5",d:"M5 15L15 5M5 5l10 10"})));var we=function({style:e,className:t}){return m.default.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",width:"54",height:"14",fill:"none",viewBox:"0 0 54 14",style:e,className:t},m.default.createElement("path",{fill:"currentColor",d:"M16.293 3.476v1.036h1.593v1.256h-1.593v5.098h-1.41V5.768H14V4.512h.883V3.244c0-.67.294-1.744 1.777-1.744.515 0 .969.049 1.361.146l-.233 1.232a5.939 5.939 0 00-.833-.073c-.442 0-.662.22-.662.67zm6.534.975V5.83c-.846 0-1.63.159-2.342.476v4.56h-1.41V4.513h1.263l.086.61c.846-.451 1.655-.67 2.403-.67zm2.505-.951c-.331.33-.944.33-1.287 0a.93.93 0 01-.246-.659c0-.268.086-.487.246-.646.343-.33.956-.33 1.287 0 .343.33.343.964 0 1.305zm.061 7.366h-1.41V4.512h1.41v6.354zm6.928-5.756c.246.146.368.402.368.756v4.976c0 1.804-.858 2.658-2.672 2.658-.92 0-1.753-.146-2.514-.439l.417-1.073c.674.22 1.336.33 1.974.33.98 0 1.385-.379 1.385-1.403v-.171c-.588.134-1.09.207-1.52.207-.907 0-1.655-.305-2.231-.902-.576-.598-.87-1.39-.87-2.354 0-.963.294-1.756.87-2.354.576-.61 1.324-.914 2.231-.914 1.005 0 1.864.232 2.562.683zm-2.488 4.634a5.15 5.15 0 001.446-.22V5.951a3.695 3.695 0 00-1.446-.292c-1.08 0-1.778.841-1.778 2.048 0 1.22.699 2.037 1.778 2.037zm7.34-5.317c1.52 0 2.28.878 2.28 2.634v3.805h-1.275l-.073-.524c-.601.414-1.288.621-2.084.621-1.263 0-2.06-.658-2.06-1.731 0-1.269 1.25-2.025 3.408-2.025.135 0 .503.013.662.013v-.171c0-1.012-.343-1.451-1.115-1.451-.675 0-1.435.158-2.256.475l-.466-1.012c1.017-.427 2.01-.634 2.979-.634zm-1.839 4.756c0 .427.343.695 1.017.695.528 0 1.251-.22 1.68-.512V8.22h-.441c-1.508 0-2.256.317-2.256.963zm9.953-4.549v-2.83h1.41v7.72c0 .354-.123.598-.368.757-.71.45-1.57.67-2.562.67-.907 0-1.655-.305-2.231-.902-.577-.61-.87-1.39-.87-2.354 0-.963.293-1.756.87-2.354.576-.61 1.324-.914 2.23-.914.43 0 .933.073 1.521.207zM43.84 9.72c.503 0 .981-.098 1.447-.293V5.854a5.15 5.15 0 00-1.447-.22c-1.078 0-1.777.817-1.777 2.037s.699 2.049 1.777 2.049zM54 7.866v.439h-4.573c.184.963.858 1.512 1.827 1.512.613 0 1.275-.146 1.986-.451l.466 1.024c-.87.378-1.729.573-2.575.573-.931 0-1.692-.304-2.268-.902-.576-.61-.87-1.402-.87-2.366 0-.975.294-1.768.87-2.366.576-.597 1.324-.902 2.244-.902.968 0 1.691.33 2.17.975.478.647.723 1.464.723 2.464zm-4.61-.586h3.298c-.086-1.073-.613-1.731-1.581-1.731-.969 0-1.582.695-1.717 1.731z"}),m.default.createElement("path",{fill:"currentColor",fillRule:"evenodd",d:"M1.196 1.229A4.009 4.009 0 014.08 0l4.092.027C9.183.027 10 .867 10 1.904c0 .6-.273 1.133-.7 1.478-.31.25-.7.399-1.126.4h-.001l-4.09-.027h-.002a4.804 4.804 0 00-2.614.77A4.986 4.986 0 000 5.974v-1.78C0 3.036.456 1.988 1.196 1.23zm4.525 4.65a4.282 4.282 0 00-1.184 2.513l3.637.023c.131 0 .259-.015.382-.042h.002c.81-.178 1.42-.908 1.44-1.788v-.046a1.9 1.9 0 00-.533-1.328 1.813 1.813 0 00-.908-.508h-.002l-.002-.001a1.68 1.68 0 00-.366-.042A4.084 4.084 0 005.72 5.88zm-4.525-.016A4.235 4.235 0 000 8.829C0 10.997 1.601 12.78 3.654 13V9.265h-.005l.005-.439v-.437h.023a5.175 5.175 0 011.439-3.13 5.05 5.05 0 01.72-.614l-1.754-.011H4.08c-.787 0-1.521.229-2.144.625a4.11 4.11 0 00-.74.603z",clipRule:"evenodd"}))},_e=g.default.div`
  ${e=>pe(e)} {
    background: ${e=>e.appearance.theme.colorBackground};
  }

  box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.06);
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
  max-width: ${e=>e.maxWidth}px;
  min-width: 300px;
  z-index: ${e=>e.zIndex};
  overflow: hidden;
`,ke=g.default.button`
  ${e=>pe(e)} {
    ${e=>e.hasImage?"\n  display: block;\n  cursor: pointer;\n  position: absolute;\n  background-color: rgba(0, 0, 0, 0.2);\n  color: #ffffff;\n  padding: 4px;\n  border-radius: 100px;\n  border-width: 0px;\n  top: 12px;\n  right: 12px;\n  box-sizing: border-box;\n  :hover {\n    opacity: 0.8;\n  }":"\n  display: block;\n  cursor: pointer;\n  position: absolute;\n  top: 12px;\n  right: 12px;\n  background-color: transparent;\n  border: none;\n  "};
  }
`,Se=g.default.img`
  ${e=>pe(e)} {
    display: block;
    width: 100%;
    height: auto;
    min-height: 200px;
    object-fit: cover;
  }
`,Ee=g.default.div`
  ${e=>pe(e)} {
    display: block;
    width: 100%;
    height: auto;
    margin-top: ${e=>e.dismissible?"24px":"0px"};
    object-fit: cover;
  }
`,Ae=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-content: center;
`,Pe=g.default.div`
  padding: 22px 22px 12px;
`,je=g.default.div`
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
`,Ne=g.default.div`
  display: flex;
  flex: 2;
  flex-shrink: 1;
  gap: 8px;
  height: 64px;
  ${e=>pe(e)} {
    flex-direction: row;
    justify-content: ${e=>e.showStepCount?"flex-end":"flex-start"};
    align-content: center;
    align-items: center;
  }
`,Ce=g.default.p`
  ${e=>pe(e)} {
    font-style: normal;
    font-weight: 600;
    font-size: 15px;
    line-height: 22px;
    color: #808080;
  }
  margin: 0;
`,Oe=g.default.div`
  background-color: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.colorBackground}};
  position: absolute;
  bottom: -47px;
  left: 0;
  width: 100%;
  height: 40px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.borderRadius}}px;
`,Te=g.default(_e)`
  background-color: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.colorBackground}};
  position: absolute;
  bottom: -60px;
  left: 0;
  width: 100%;
  height: 40px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.borderRadius}}px;
  padding: 0;
  z-index: ${e=>e.zIndex};
`,Ie=g.default.div`
  display: flex;
  justify-content: center;
  align-items: center;
  font-style: normal;
  font-weight: 500;
  font-size: 12px;
  line-height: 18px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
`;function Me({appearance:e}){return m.default.createElement(Ie,{className:ue("poweredByFrigadeContainer",e),appearance:e},"Powered by  ",m.default.createElement(we,null))}var De=g.default.div`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background-color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBackground}};
    /* Mobile */
    @media (max-width: 500px) {
      width: 90%;
      height: 90%;
      top: 50%;
      left: 50%;
    }

    width: ${e=>{var t;return null!=(t=e.width)?t:"1000px"}};
    z-index: 1500;
    border-radius: ${e=>{var t,n,r;return null!=(r=null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.borderRadius)?r:8}}px;
    ${e=>fe(e)}
  }

  padding: 32px;

  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  max-height: 90%;

  display: flex;
  flex-direction: column;
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  animation-duration: 0.15s;
  animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
  animation-name: fadeIn;
  box-sizing: border-box;

  @keyframes fadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
`,Le=g.default.div`
  position: relative;
  flex: 0 1 auto;
`,$e=g.default.div`
  position: absolute;
  top: 16px;
  right: 16px;
  cursor: pointer;
  z-index: 1501;
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  }
`,Re=g.default.div`
  overflow: scroll;
  flex: 1 1;
  display: flex;
  ::-webkit-scrollbar {
    display: none;
  }
`,Fe=({onClose:e,visible:t,headerContent:n=null,style:a=null,children:i,appearance:o,dismissible:s=!0,showFrigadeBranding:l=!1})=>{let[d,u]=r.useState("");return r.useEffect((()=>{let e=getComputedStyle(document.body).getPropertyValue("overflow");u(e)}),[]),r.useEffect((()=>{let t=t=>{"Escape"===t.key&&e()};return document.addEventListener("keydown",t),()=>{document.removeEventListener("keydown",t)}}),[e]),r.useEffect((()=>{let e=document.body.style;return t?e.setProperty("overflow","hidden"):e.setProperty("overflow",d),()=>{e.setProperty("overflow",d)}}),[t]),t?m.default.createElement(c.Portal,null,m.default.createElement(ve,{appearance:o,onClose:()=>{s&&e()}}),m.default.createElement(De,{appearance:o,className:ue("modalContainer",o),styleOverrides:a},s&&m.default.createElement($e,{className:ue("modalClose",o),onClick:()=>e(),appearance:o},m.default.createElement(xe,null)),n&&m.default.createElement(Le,null,n),m.default.createElement(Re,null,i),l&&m.default.createElement(Oe,{appearance:o,className:ue("poweredByFrigadeRibbon",o)},m.default.createElement(Me,{appearance:o})))):m.default.createElement(m.default.Fragment,null)};var Ue=g.default.div`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBackground}};
    position: fixed;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    z-index: 1500;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    width: 350px;
    padding: 24px;
  }
  ${e=>function(e){switch(e){case"top-left":return"\n        top: 0;\n        left: 0;\n      ";case"top-right":return"\n        top: 0;\n        right: 0;\n      ";case"bottom-left":return"\n        bottom: 0;\n        left: 0;\n      "}return"right: 0; bottom: 0;"}(e.modalPosition)}
  margin: 28px;
`,Be=g.default.div`
  position: relative;
  flex: 1;
`,ze=g.default.div`
  position: absolute;
  top: 16px;
  right: 16px;
  cursor: pointer;
  z-index: 1501;
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  }
`,Ve=g.default.div`
  overflow: scroll;
  flex: 5;
  ::-webkit-scrollbar {
    display: none;
  }
`,He=({onClose:e,visible:t,headerContent:n=null,children:a,appearance:i,modalPosition:o="bottom-right",dismissible:s=!0})=>{let[l,d]=r.useState("");return r.useEffect((()=>{let e=getComputedStyle(document.body).getPropertyValue("overflow");d(e)}),[]),r.useEffect((()=>{let t=t=>{"Escape"===t.key&&e()};return document.addEventListener("keydown",t),()=>{document.removeEventListener("keydown",t)}}),[e]),r.useEffect((()=>{let e=document.body.style;return t?e.setProperty("overflow","hidden"):e.setProperty("overflow",l),()=>{e.setProperty("overflow",l)}}),[t]),t?m.default.createElement(c.Portal,null,m.default.createElement(Ue,{appearance:i,className:ue("cornerModalContainer",i),modalPosition:o},s&&m.default.createElement(ze,{className:ue("cornerModalClose",i),onClick:()=>e()},m.default.createElement(xe,null)),n&&m.default.createElement(Be,null,n),m.default.createElement(Ve,null,a))):m.default.createElement(m.default.Fragment,null)};function Ye(){let{defaultAppearance:e}=r.useContext(Yn);return{mergeAppearanceWithDefault:function(t){var n,r,a;let i=JSON.parse(JSON.stringify(e));return t?{styleOverrides:Object.assign(null!=(n=i.styleOverrides)?n:{},null!=(r=t.styleOverrides)?r:{}),theme:Object.assign(i.theme,null!=(a=t.theme)?a:{})}:i}}}var We=g.default.label`
  ${e=>pe(e)} {
    font-size: 12px;
    line-height: 18px;
    margin-bottom: 5px;
    margin-top: 10px;
    font-style: normal;
    font-weight: 600;
    letter-spacing: 0.24px;
  }
  display: flex;
`,qe=g.default.label`
  ${e=>pe(e)} {
    font-size: 12px;
    line-height: 20px;
    margin-bottom: 5px;
  }
  display: flex;
`,Ke=g.default.span`
  font-weight: 400;
  font-size: 14px;
  line-height: 22px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextError}};
  display: flex;
  margin-right: 5px;
  margin-top: 10px;
`,Ge=g.default.div`
  display: flex;
  align-items: flex-start;
  justify-content: left;
  margin-bottom: 10px;
`,Ze={theme:{colorPrimary:"#0171F8",colorSecondary:"#2E343D",colorText:"#0F1114",colorBackground:"#ffffff",colorBackgroundSecondary:"#d2d2d2",colorTextOnPrimaryBackground:"#ffffff",colorTextSecondary:"#2E343D",colorTextDisabled:"#5A6472",colorBorder:"#E5E5E5",colorTextError:"#c00000",colorTextSuccess:"#00D149",borderRadius:10}};function Je({title:e,required:t,appearance:n=Ze}){return e?m.default.createElement(Ge,{className:ue("formLabelWrapper",n)},t?m.default.createElement(Ke,{className:ue("formLabelRequired",n),appearance:n},"*"):null,m.default.createElement(We,{className:ue("formLabel",n)},e)):null}function Xe({title:e,appearance:t}){return e?m.default.createElement(Ge,null,m.default.createElement(qe,{className:ue("formSubLabel",t)},e)):null}var Qe=g.default.div`
  display: flex;
  flex-direction: column;
  width: 100%;
`,et=g.default.input`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    border: 1px solid ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
    font-size: 14px;
    ::placeholder {
      color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextDisabled}};
      font-size: 14px;
    }
    border-radius: 6px;
  }
  width: 100%;
  height: 40px;
  box-sizing: border-box;
  padding: 0 10px;
  margin-bottom: 10px;
`,tt=g.default.textarea`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    border: 1px solid ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
    font-size: 14px;
    padding: 10px;
    ::placeholder {
      color: #c7c7c7;
      font-size: 14px;
    }
    border-radius: 6px;
  }
  width: 100%;
  min-height: 70px;
  box-sizing: border-box;
  margin-bottom: 10px;
  resize: none;
`;function nt({formInput:e,customFormTypeProps:t,onSaveInputData:n,setFormValidationErrors:a,inputData:i}){let o=e,[s,l]=r.useState((null==i?void 0:i.text)||""),[c,u]=r.useState(!1),[p,h]=r.useState(!1),f=et;function g(e){var t;if(l(e),n({text:e}),!0===o.required&&""===e.trim())return void a([{id:o.id,message:`${null!=(t=o.title)?t:"Field"} is required`,hidden:!p}]);let r=function(e,t){var n,r,a,i,o,s,l;try{if(t){if("number"==t.type){let o=d.z.number();if(t.props)for(let e of t.props)"min"==e.requirement?o=o.min(Number(e.value),null!=(n=e.message)?n:"Value is too small"):"max"==e.requirement?o=o.max(Number(e.value),null!=(r=e.message)?r:"Value is too large"):"positive"==e.requirement?o=o.positive(null!=(a=e.message)?a:"Value must be positive"):"negative"==e.requirement&&(o=o.nonpositive(null!=(i=e.message)?i:"Value must be negative"));o.parse(Number(e))}if("string"==t.type){let n=d.z.string();if(t.props)for(let e of t.props)"min"==e.requirement?n=n.min(Number(e.value),null!=(o=e.message)?o:"Value is too short"):"max"==e.requirement?n=n.max(Number(e.value),null!=(s=e.message)?s:"Value is too long"):"regex"==e.requirement&&(n=n.regex(new RegExp(String(e.value)),null!=(l=e.message)?l:"Value does not match requirements"));n.parse(e)}return}}catch(e){if(e instanceof d.z.ZodError)return e.issues&&e.issues.length>0?e.issues[0].message:null}return null}(e,o.validation);!r||""===e.trim()&&!0!==o.required?a([]):a([{id:o.id,message:r,hidden:!p}])}return r.useEffect((()=>{""===s&&!c&&(u(!0),g(""))}),[]),r.useEffect((()=>{p&&g(s)}),[p]),o.multiline&&(f=tt),m.default.createElement(Qe,null,m.default.createElement(Je,{title:o.title,required:o.required,appearance:t.appearance}),m.default.createElement(f,{className:ue("inputComponent",t.appearance),value:s,onChange:e=>{g(e.target.value)},appearance:t.appearance,placeholder:o.placeholder,type:function(){var e;switch(null==(e=null==o?void 0:o.validation)?void 0:e.type){case"email":return"email";case"number":return"number";case"password":return"password"}return null}(),onBlur:()=>{h(!0)}}),m.default.createElement(Xe,{title:o.subtitle,appearance:t.appearance}))}var rt=g.default.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
`,at=g.default.select`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    border: 1px solid ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
    font-size: 14px;
    border-radius: 6px;
  }
  width: 100%;
  height: 40px;
  box-sizing: border-box;

  padding: 0 10px;
  margin-bottom: 10px;
  color: ${e=>{var t,n,r,a;return""==e.value?null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextDisabled:null==(a=null==(r=e.appearance)?void 0:r.theme)?void 0:a.colorText}};

  appearance: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'><path stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/></svg>");
  background-position: right 0.5rem center;
  background-repeat: no-repeat;
  background-size: 1.5em 1.5em;
  -webkit-print-color-adjust: exact;
`;function it({formInput:e,customFormTypeProps:t,onSaveInputData:n,inputData:a,setFormValidationErrors:i}){var o,s,l,c,d,u,p,h,f;let g=e,[y,v]=r.useState(!1),[b,x]=r.useState(!1);return r.useState(""),r.useEffect((()=>{var e,t,r,i,o;if(!(null!=(e=null==a?void 0:a.choice)&&e[0]||y)){if(v(!0),g.requireSelection)return void n({choice:[""]});if(g.defaultValue&&null!=(t=g.props.options)&&t.find((e=>e.id===g.defaultValue))){let e=null==(r=g.props.options)?void 0:r.find((e=>e.id===g.defaultValue));n({choice:[e.id],label:[e.title]})}else n({choice:[(null==(i=g.props.options)?void 0:i[0].id)||""],label:[null==(o=g.props.options)?void 0:o[0].title]})}}),[]),r.useEffect((()=>{var e;g.requireSelection&&""===(null==(e=null==a?void 0:a.choice)?void 0:e[0])?i([{message:"Please select an option",id:g.id,hidden:!0}]):i([])}),[null==(o=null==a?void 0:a.choice)?void 0:o[0],b]),m.default.createElement(rt,null,m.default.createElement(Je,{title:g.title,required:g.required,appearance:t.appearance}),m.default.createElement(at,{value:null==(s=null==a?void 0:a.choice)?void 0:s[0],onChange:e=>{x(!0),n({choice:[e.target.value],label:[e.target.selectedOptions[0].text]})},placeholder:g.placeholder,appearance:t.appearance,className:ue("multipleChoiceSelect",t.appearance)},g.requireSelection&&m.default.createElement("option",{key:"null-value",value:"",disabled:!0},null!=(l=g.placeholder)?l:"Select an option"),null==(c=g.props.options)?void 0:c.map((e=>m.default.createElement("option",{key:e.id,value:e.id},e.title)))),(null==(u=null==(d=g.props.options)?void 0:d.find((e=>{var t;return e.id===(null==(t=null==a?void 0:a.choice)?void 0:t[0])})))?void 0:u.isOpenEnded)&&m.default.createElement(m.default.Fragment,null,m.default.createElement(Je,{title:null!=(f=null==(h=null==(p=g.props.options)?void 0:p.find((e=>{var t;return e.id===(null==(t=null==a?void 0:a.choice)?void 0:t[0])})))?void 0:h.openEndedLabel)?f:"Please specify",required:!1,appearance:t.appearance}),m.default.createElement(et,{type:"text",placeholder:"Enter your answer here",onChange:e=>{var t,r;n({choice:[null==(r=null==(t=g.props.options)?void 0:t.find((e=>{var t;return e.id===(null==(t=null==a?void 0:a.choice)?void 0:t[0])})))?void 0:r.id],label:[e.target.value],isOpenEnded:!0})},appearance:t.appearance})),m.default.createElement(Xe,{title:g.subtitle,appearance:t.appearance}))}var ot=({color:e,percentage:t,size:n})=>{let r=.5*n-2,a=2*Math.PI*r,i=(1-t)*a;return m.default.createElement("circle",{r,cx:.5*n,cy:.5*n,fill:"transparent",stroke:i!==a?e:"",strokeWidth:"3px",strokeDasharray:a,strokeDashoffset:t?i:0})},st=({fillColor:e,size:t,percentage:n,children:r,bgColor:a="#D9D9D9",className:i,style:o})=>m.default.createElement("svg",{style:o,className:i,width:t,height:t,overflow:"visible"},m.default.createElement("g",{transform:`rotate(-90 ${.5*t} ${.5*t})`},m.default.createElement(ot,{color:a,size:t}),m.default.createElement(ot,{color:e,percentage:Math.max(n,.1),size:t})),r),lt=({color:e="#FFFFFF"})=>m.default.createElement("svg",{width:10,height:8,viewBox:"0 0 10 8",fill:"none",xmlns:"http://www.w3.org/2000/svg"},m.default.createElement("path",{d:"M1 4.34815L3.4618 7L3.4459 6.98287L9 1",stroke:e,strokeWidth:"1.5",strokeLinecap:"round",strokeLinejoin:"round"})),ct={width:"22px",height:"22px",borderRadius:"8px",display:"flex",justifyContent:"center",alignItems:"center"},dt={width:"22px",height:"22px",borderRadius:"40px",display:"flex",justifyContent:"center",alignItems:"center"},ut={border:"1px solid #000000",color:"#FFFFFF"},pt={border:"1px solid #C5CBD3"},ht={color:"#FFFFFF"},ft={border:"1px solid #C5CBD3"},mt=g.default.div`
  ${e=>fe(e)}
  flex-shrink: 0;
`,gt=({value:e,type:t="round",primaryColor:n="#000000",progress:r,appearance:a=Ze,style:i,className:o})=>{let s=(e=>"square"===e?ct:dt)(t),l=((e,t)=>"square"===e?t?ut:pt:t?ht:ft)(t,e);return s=!0===e?N(j(j({},s),l),{backgroundColor:a.theme.colorTextSuccess,borderColor:"square"===t?n:"none"}):j(j({},s),l),!0!==e&&"round"===t&&void 0!==r&&1!==r?m.default.createElement(st,{fillColor:n,percentage:r,size:22}):m.default.createElement(mt,{styleOverrides:s,style:i,role:"checkbox",className:me(ue("checkIconContainer",a),ue(e?"checkIconContainerChecked":"checkIconContainerUnchecked",a),e?"checkIconContainerChecked":"checkIconContainerUnchecked",o)},e&&m.default.createElement(lt,{color:"#FFFFFF"}))};function yt(e){return e?{__html:x.default.sanitize(e,{ALLOWED_TAGS:["b","i","a","span","div","p","pre","u","br","img","code","li","ul","table","tbody","thead","tr","td","th","h1","h2","h3","h4","video"],ALLOWED_ATTR:["style","class","target","id","href","alt","src","controls","autoplay","loop","muted"]})}:{__html:""}}var vt=g.default.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
`,bt=g.default.button`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    border: 1px solid ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
    font-size: 14px;
    // Selector for when selected=true
    &[data-selected='true'] {
      border: 1px solid ${e=>e.appearance.theme.colorPrimary};
      background-color: ${e=>e.appearance.theme.colorPrimary}1a;
    }

    :hover {
      border: 1px solid ${e=>e.appearance.theme.colorPrimary};
    }
    text-align: left;
    border-radius: 10px;
  }
  display: inline-flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  width: 100%;
  line-height: 18px;
  padding: 18px;
  margin-bottom: 10px;
`;var xt=g.default.h1`
  ${e=>pe(e)} {
    font-style: normal;
    font-weight: 700;
    font-size: ${e=>"small"==e.size?"15px":"18px"};
    line-height: ${e=>"small"==e.size?"22px":"24px"};
    letter-spacing: 0.36px;
    display: flex;
    align-items: center;
    margin-bottom: 4px;
    color: ${e=>e.appearance.theme.colorText};
  }
`,wt=g.default.h2`
  ${e=>pe(e)} {
    font-style: normal;
    font-weight: 400;
    font-size: 14px;
    line-height: 22px;
    letter-spacing: 0.28px;
    color: ${e=>e.appearance.theme.colorTextSecondary};
  }
`;function _t({appearance:e,title:t,subtitle:n,size:r="medium",classPrefix:a="",ariaPrefix:i=""}){return m.default.createElement(m.default.Fragment,null,m.default.createElement(xt,{appearance:e,id:i?`frigade${i}Title`:"frigadeTitle",className:ue(`${a}${a?ge(r):r}Title`,e),dangerouslySetInnerHTML:yt(t),size:r}),n&&m.default.createElement(wt,{id:i?`frigade${i}Subtitle`:"frigadeSubtitle",appearance:e,className:ue(`${a}${a?ge(r):r}Subtitle`,e),dangerouslySetInnerHTML:yt(n),size:r}))}var kt=e=>m.createElement("svg",j({xmlns:"http://www.w3.org/2000/svg",width:12,height:12,"aria-hidden":"true",viewBox:"0 0 16 16"},e),m.createElement("path",{fill:"currentColor",fillRule:"evenodd",d:"m10.115 1.308 5.635 11.269A2.365 2.365 0 0 1 13.634 16H2.365A2.365 2.365 0 0 1 .25 12.577L5.884 1.308a2.365 2.365 0 0 1 4.231 0zM8 10.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM8 9c.552 0 1-.32 1-.714V4.714C9 4.32 8.552 4 8 4s-1 .32-1 .714v3.572C7 8.68 7.448 9 8 9z"})),St=g.default.div`
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  overflow: visible;
  padding-top: 14px;
`,Et=g.default.div`
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextError}};
  font-size: 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
`,At=g.default.div`
  margin-right: 4px;
  display: inline-flex;
`,Pt=g.default.div`
  padding-left: 1px;
  padding-right: 1px;
`,jt={text:nt,multipleChoice:it,multipleChoiceList:function({formInput:e,customFormTypeProps:t,onSaveInputData:n,inputData:a,setFormValidationErrors:i}){var o;let s=e,[l,c]=r.useState((null==a?void 0:a.choice)||[]),[d,u]=r.useState(!1);return r.useEffect((()=>{0==l.length&&!d&&(u(!0),n({choice:[]}))}),[]),r.useEffect((()=>{n({choice:l})}),[l]),r.useEffect((()=>{s.required&&(l.length<s.props.minChoices||l.length>s.props.maxChoices)?i([{message:"",id:s.id}]):i([])}),[l]),m.default.createElement(vt,null,m.default.createElement(Je,{title:s.title,required:s.required,appearance:t.appearance}),null==(o=s.props.options)?void 0:o.map((e=>m.default.createElement(bt,{appearance:t.appearance,className:ue(l.includes(e.id)?"multipleChoiceListItemSelected":"multipleChoiceListItem",t.appearance),key:e.id,value:e.id,"data-selected":l.includes(e.id),onClick:()=>{l.includes(e.id)?c(l.filter((t=>t!==e.id))):l.length<s.props.maxChoices?c([...l,e.id]):1==l.length&&1==s.props.maxChoices&&c([e.id])}},m.default.createElement("span",{dangerouslySetInnerHTML:yt(e.title)}),m.default.createElement(gt,{type:"round",primaryColor:t.appearance.theme.colorPrimary,value:l.includes(e.id),appearance:t.appearance})))),m.default.createElement(Xe,{title:s.subtitle,appearance:t.appearance}))}},Nt="frigade-multiInputStepTypeData";function Ct({flowId:e,stepData:t,canContinue:n,setCanContinue:a,onSaveData:i,appearance:o,customFormElements:s,prefillData:l}){var c;let d=t.props,[u,p]=r.useState([]),[h,f]=r.useState([]),{userId:g}=le(),[y,v]=r.useState(function(){if("undefined"!=typeof window&&window.localStorage){let e=window.localStorage.getItem(w());if(e)return JSON.parse(e)}return null}()||(l?l[t.id]:null)||{}),{readonly:b}=r.useContext(Yn),x=j(j({},jt),s);function w(){return`${Nt}-${e}-${t.id}-${g}`}return r.useEffect((()=>{a(0===u.length)}),[u,a]),r.useEffect((()=>{i(y)}),[y]),m.default.createElement(Pt,{className:ue("multiInput",o)},m.default.createElement(_t,{appearance:o,title:t.title,subtitle:t.subtitle}),m.default.createElement(St,{className:ue("multiInputContainer",o)},null==(c=null==d?void 0:d.data)?void 0:c.map((r=>{let s=u.reverse().find((e=>e.id===r.id));return x[r.type]?m.default.createElement("span",{key:r.id,"data-field-id":r.id,className:ue("multiInputField",o)},x[r.type]({formInput:r,customFormTypeProps:{flowId:e,stepData:t,canContinue:n,setCanContinue:a,onSaveData:i,appearance:o},onSaveInputData:e=>{!h.includes(r.id)&&e&&""!==(null==e?void 0:e.text)&&f((e=>[...e,r.id])),function(e,t){v((n=>{let r=N(j({},n),{[e.id]:t});return"undefined"!=typeof window&&window.localStorage&&!b&&window.localStorage.setItem(w(),JSON.stringify(r)),r}))}(r,e)},inputData:y[r.id],allInputData:y,setFormValidationErrors:e=>{0===e.length&&0===u.length||p((t=>0===e.length?t.filter((e=>e.id!==r.id)):[...t,...e]))}}),s&&s.message&&h.includes(r.id)&&!0!==s.hidden&&m.default.createElement(Et,{key:r.id,style:{overflow:"hidden"},appearance:o,className:ue("multiInputValidationError",o)},m.default.createElement(At,{appearance:o,className:ue("multiInputValidationErrorIcon",o)},m.default.createElement(kt,null)),s.message)):null}))))}var Ot=g.default.div`
  align-items: center;
  display: flex;
  justify-content: ${e=>e.showBackButton?"space-between":"flex-end"};
  padding-top: 14px;
`,Tt=g.default.div`
  color: ${e=>e.appearance.theme.colorTextError};
  font-size: 12px;
`,It=g.default.div`
  display: flex;
  gap: 12px;
  width: 100%;
  justify-content: flex-end;
`,Mt=g.default.div`
  display: flex;
  // If type is set to large-modal, use padding 60px horizontal, 80px vertical
  // Otherwise, use 4px padding
  flex-direction: column;
  flex-grow: 1;
  flex-basis: 0;
  position: relative;
`,Dt=g.default.div`
  padding: ${e=>"large-modal"===e.type?"50px":"0px"};
  position: relative;
  overflow-y: auto;
`,Lt=g.default.div`
  display: flex;
  align-self: stretch;
  flex-grow: 1;
  flex-basis: 0;
  // If props.image is set, use it as the background image
  background-image: ${e=>e.image?`url(${e.image})`:"none"};
  // scale background image to fit
  background-size: contain;
  background-position: center;
  border-top-right-radius: ${e=>e.appearance.theme.borderRadius}px;
  border-bottom-right-radius: ${e=>e.appearance.theme.borderRadius}px;
`,$t=g.default.div`
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255, 255, 255, 0.25);
  border-bottom-color: #fff;
  border-radius: 50%;
  display: inline-block;
  box-sizing: border-box;
  animation: rotation 0.75s linear infinite;

  @keyframes rotation {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`,Rt=g.default.button`
  justify-content: center;
  align-content: center;
  align-items: center;
  ${e=>pe(e)} {
    display: flex;
    // Anything inside this block will be ignored if the user provides a custom class
    width: ${e=>"full-width"===e.type?"100%":"auto"};
    // Only add margin if prop withMargin is true
    ${e=>e.withMargin?"margin: 16px 0px 16px 0px;":""}

    border: 1px solid ${e=>{var t,n;return e.secondary?"#C5CBD3":null==(n=null==(t=null==e?void 0:e.appearance)?void 0:t.theme)?void 0:n.colorPrimary}};
    color: ${e=>{var t,n,r,a;return e.secondary?null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText:null==(a=null==(r=e.appearance)?void 0:r.theme)?void 0:a.colorTextOnPrimaryBackground}};
    background-color: ${e=>{var t,n,r,a;return e.secondary?null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBackground:null==(a=null==(r=null==e?void 0:e.appearance)?void 0:r.theme)?void 0:a.colorPrimary}};
    border-radius: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.borderRadius}}px;
    padding: ${e=>"small"==e.size?"6px 14px 6px 14px":"8px 20px 8px 20px"};
    font-size: ${e=>"small"==e.size?"14px":"15px"};
    line-height: 20px;
    font-weight: 500;
    ${e=>fe(e)}
  }

  font-family: inherit;

  cursor: pointer;
  :hover {
    opacity: ${e=>"true"==e.loading?"1.0":"0.8"};
  }

  :disabled {
    opacity: ${e=>"true"==e.loading?"1.0":"0.3"};
    cursor: not-allowed;
  }
`,Ft=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    margin-top: 8px;

    & > * {
      margin-right: 8px;
    }
  }
`,Ut=({onClick:e,title:t,style:n,disabled:r,type:a="inline",size:i="medium",secondary:o=!1,appearance:s,withMargin:l=!0,classPrefix:c="",loading:d=!1})=>{var u;let p={tabindex:o?"0":"1",secondary:o,appearance:s,disabled:r||d,loading:null!=(u=null==d?void 0:d.toString())?u:"",onClick:e,styleOverrides:n,type:a,withMargin:l,size:i,className:ue(function(){let e=o?"buttonSecondary":"button";return""===c?e:`${c}${ge(e)}`}(),s)};return d?m.default.createElement(Rt,j({},p),m.default.createElement($t,{className:ue("buttonLoader",s)})):m.default.createElement(Rt,N(j({},p),{dangerouslySetInnerHTML:yt(null!=t?t:"Continue")}))},Bt=({step:e,canContinue:t,appearance:n,onPrimaryClick:r,onSecondaryClick:a,selectedStep:i,steps:o,onBack:s,allowBackNavigation:l,errorMessage:c,isSaving:d})=>{var u;let p=o.length>1&&0!=i&&l,h=e.primaryButtonTitle&&e.secondaryButtonTitle||p?"inline":"full-width";return m.default.createElement(m.default.Fragment,null,null!==c&&null!=c&&m.default.createElement(Tt,{appearance:n,className:ue("formCTAError",n)},c),m.default.createElement(Ot,{showBackButton:p,className:ue("formCTAContainer",n)},p&&m.default.createElement(Ut,{title:null!=(u=e.backButtonTitle)?u:"Back",onClick:s,secondary:!0,withMargin:!1,type:h,appearance:n,style:{width:"90px",maxWidth:"90px"},classPrefix:"back"}),m.default.createElement(It,{className:ue("ctaWrapper",n)},e.secondaryButtonTitle?m.default.createElement(Ut,{title:e.secondaryButtonTitle,onClick:a,secondary:!0,withMargin:!1,type:h,appearance:n,disabled:d}):null," ",e.primaryButtonTitle?m.default.createElement(Ut,{disabled:!t,withMargin:!1,title:e.primaryButtonTitle,onClick:r,type:h,appearance:n,loading:d}):null)))},zt=g.default.div`
  text-align: center;
  color: #e6e6e6;
`,Vt=({stepCount:e=0,currentStep:t=0,className:n,appearance:r})=>{let{theme:a}=Ye().mergeAppearanceWithDefault(r);return m.default.createElement(zt,{className:n},m.default.createElement("svg",{width:16*e-8,height:8,viewBox:`0 0 ${16*e-8} 8`,fill:"none"},Array(e).fill(null).map(((e,n)=>m.default.createElement("rect",{key:n,x:16*n,y:0,width:8,height:8,rx:4,fill:t===n?a.colorPrimary:"currentColor"})))))};function Ht(){let e=r.useContext(Yn);function t(t,n){if(!t)return;let r=t.startsWith("http")?"_blank":"_self";n&&"_blank"!==n&&(r="_self"),e.navigate(t,r)}return{primaryCTAClickSideEffects:function(e){t(e.primaryButtonUri,e.primaryButtonUriTarget)},secondaryCTAClickSideEffects:function(e){t(e.secondaryButtonUri,e.secondaryButtonUriTarget)},handleUrl:t}}var Yt=g.default.div`
  display: flex;
  flex-wrap: wrap;
  align-content: center;
  justify-content: center;
`,Wt=g.default.div`
  align-content: center;
  align-items: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  flex-grow: 1;
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  margin: 15px;
  padding: 20px;
  flex-basis: 255px;
  flex-grow: 0;
  flex-shrink: 0;
`,qt=g.default.img`
  width: 78px;
  height: auto;
`,Kt=g.default.button`
  font-style: normal;
  font-weight: 600;
  font-size: 13px;
  line-height: 16px;

  display: flex;
  align-items: center;
  text-align: center;
  border: 1px solid;
  border-radius: 100px;
  padding: 8px 12px;
  margin-top: 16px;
`,Gt=g.default.h1`
  font-weight: 700;
  font-size: 28px;
  line-height: 34px;
`,Zt=g.default.h2`
  font-style: normal;
  font-weight: 400;
  font-size: 16px;
  line-height: 24px;
  color: #7e7e7e;
  margin-top: 12px;
  margin-bottom: 16px;
  max-width: 70%;
`;function Jt({stepData:e,appearance:t}){var n,r;let{handleUrl:a}=Ht();return m.default.createElement("div",null,m.default.createElement(Gt,{dangerouslySetInnerHTML:yt(e.title)}),m.default.createElement(Zt,{dangerouslySetInnerHTML:yt(e.subtitle)}),m.default.createElement(Yt,null,null==(r=null==(n=e.props)?void 0:n.links)?void 0:r.map((e=>m.default.createElement(Wt,{key:e.title},m.default.createElement(qt,{src:e.imageUri}),m.default.createElement(Kt,{style:{borderColor:t.theme.colorPrimary,color:t.theme.colorPrimary},onClick:()=>{var t;e.uri&&a(e.uri,null!=(t=e.uriTarget)?t:"_blank")}},e.title))))))}var Xt=({style:e,className:t})=>m.default.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:1.5,stroke:"currentColor",className:t,style:e},m.default.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}),m.default.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M15.91 11.672a.375.375 0 010 .656l-5.603 3.113a.375.375 0 01-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112z"})),Qt=g.default.div`
  display: flex;
  align-items: center;
  justify-content: flex-start;
  flex-direction: column;
  width: 100%;
  height: 100%;
  position: relative;
`,en=g.default.div`
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  cursor: pointer;
  :hover {
    opacity: 0.6;
  }
  z-index: 10;

  > svg {
    width: 40px;
    height: 40px;
    color: ${e=>e.appearance.theme.colorBackground};
    box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.06);
    border-radius: 50%;
  }
`,tn=g.default.video`
  width: 100%;
  height: 100%;
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`,nn=g.default.iframe`
  width: 100%;
  height: 100%;
  min-height: 260px;
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`,rn=g.default.iframe`
  width: 100%;
  height: 100%;
  min-height: 400px;
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`,an=g.default.iframe`
  width: 100%;
  height: 100%;
  min-height: 400px;
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`;function on({appearance:e,videoUri:t,autoplay:n=!1,loop:a=!1,hideControls:i=!1}){let o=r.useRef(),[s,l]=r.useState(n);if(t.includes("youtube")){let n=t.split("v=")[1],r=n.indexOf("&");return-1!==r&&(n=n.substring(0,r)),m.default.createElement(nn,{width:"100%",height:"100%",src:`https://www.youtube.com/embed/${n}`,frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0,appearance:e,className:ue("youtubePlayer",e)})}if(t.includes("vimeo")){let n=t.split("vimeo.com/")[1],r=n.indexOf("&");return-1!==r&&(n=n.substring(0,r)),m.default.createElement(rn,{width:"100%",height:"100%",src:`https://player.vimeo.com/video/${n}`,frameBorder:"0",allow:"autoplay; fullscreen; picture-in-picture",allowFullScreen:!0,appearance:e,className:ue("vimeoPlayer",e)})}if(t.includes("wistia")){let n=t.split("wistia.com/medias/")[1],r=n.indexOf("&");return-1!==r&&(n=n.substring(0,r)),m.default.createElement(an,{width:"100%",height:"100%",src:`https://fast.wistia.net/embed/iframe/${n}`,frameBorder:"0",allow:"autoplay; fullscreen; picture-in-picture",allowFullScreen:!0,appearance:e,className:ue("wistiaPlayer",e)})}return m.default.createElement(Qt,{className:ue("videoPlayerWrapper",e),appearance:e},!s&&m.default.createElement(en,{onClick:()=>{l(!0),o.current.play()},appearance:e,className:ue("playIconWrapper",e)},m.default.createElement(Xt,null)),m.default.createElement(tn,{appearance:e,controls:s&&!i,ref:o,play:s,src:t,autoPlay:n,muted:n,loop:a}))}var sn=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
`,ln=g.default.img`
  ${e=>pe(e)} {
    width: 100%;
    height: auto;
    max-height: 250px;
    margin-bottom: 24px;
  }
`,cn=g.default.div`
  ${e=>pe(e)} {
    margin-bottom: 24px;
  }
`,dn=g.default.div`
  ${e=>pe(e)} {
    width: 100%;
    height: auto;
    max-height: 250px;
    margin-bottom: 24px;
  }
`;function un({stepData:e,appearance:t,setCanContinue:n}){var a,i,o;return r.useEffect((()=>{n(!0)}),[]),m.default.createElement(sn,{className:ue("callToActionContainer",t)},m.default.createElement(cn,{className:ue("callToActionTextContainer",t)},m.default.createElement(_t,{appearance:t,title:e.title,subtitle:e.subtitle})),e.imageUri&&m.default.createElement(ln,{className:ue("callToActionImage",t),src:e.imageUri}),!e.imageUri&&e.videoUri&&m.default.createElement(dn,{appearance:t,className:ue("callToActionVideo",t)},m.default.createElement(on,{appearance:t,videoUri:e.videoUri,autoplay:null==(a=e.props)?void 0:a.autoplayVideo,loop:null==(i=e.props)?void 0:i.loopVideo,hideControls:null==(o=e.props)?void 0:o.hideVideoControls})))}var pn=g.default.div`
  width: auto;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 4px;
`,hn=g.default.div`
  width: 100%;
  text-align: left;
`,fn=g.default.h1`
  font-style: normal;
  font-weight: 700;
  font-size: 32px;
  line-height: 38px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
`,mn=g.default.h1`
  font-style: normal;
  font-weight: 400;
  font-size: 16px;
  line-height: 27px;
  margin-top: 16px;
  margin-bottom: 16px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
`,gn=g.default.div`
  padding-top: 12px;
  padding-bottom: 12px;
  flex-direction: row;
  display: flex;
  justify-content: space-between;
  align-items: center;
  align-content: center;
  cursor: pointer;
  border-bottom: ${e=>e.hideBottomBorder?"none":"1px solid #D8D8D8"};
  width: 100%;
`,yn=g.default.div`
  padding-top: 10px;
  padding-bottom: 10px;
  flex-direction: row;
  display: flex;
  justify-content: flex-start;
`,vn=g.default.img`
  width: 42px;
  height: 42px;
  margin-right: 12px;
`,bn=g.default.p`
  font-style: normal;
  font-weight: 500;
  font-size: 17px;
  line-height: 21px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  display: flex;
  align-self: center;
`,xn=({stepData:e,setCanContinue:t,onSaveData:n,appearance:a})=>{let i=e.props,[o,s]=r.useState([]),[l,c]=r.useState(!1),[d,u]=r.useState(e.id);return r.useEffect((()=>{0==o.length&&!l&&(c(!0),n({choice:[]}))}),[l]),r.useEffect((()=>{d!==e.id&&(u(e.id),s([]))}),[e]),r.useEffect((()=>{n({choice:o}),o.length>=i.minChoices?t(!0):t(!1)}),[o]),m.default.createElement(pn,{className:ue("selectListContainer",a)},m.default.createElement(hn,null,m.default.createElement(fn,{className:ue("selectListTitle",a)},e.title),m.default.createElement(mn,{appearance:a,className:ue("selectListSubtitle",a)},e.subtitle)),i.options.map(((e,t)=>{let n=o.includes(e.id);return m.default.createElement(gn,{key:`select-item-${t}`,onClick:()=>{o.includes(e.id)?s(o.filter((t=>t!==e.id))):o.length<i.maxChoices?s([...o,e.id]):1==o.length&&1==i.maxChoices&&s([e.id])},hideBottomBorder:t===i.options.length-1,className:ue("selectListSelectItem",a)},m.default.createElement(yn,{className:ue("selectListItemImage",a)},e.imageUri&&m.default.createElement(vn,{src:e.imageUri,alt:`select-icon-${t}`}),m.default.createElement(bn,{appearance:a,className:ue("selectListSelectItemText",a)},e.title)),m.default.createElement(gt,{appearance:a,value:n,primaryColor:a.theme.colorPrimary}))})))},wn=({children:e,id:t,shouldWrap:n=!1})=>m.default.createElement(m.default.Fragment,null,n?m.default.createElement("div",{key:t,style:{width:"100%",height:"100%",position:"absolute",top:0,left:0,zIndex:1,overflowY:"auto"}},e):e),_n=({appearance:e,steps:t,selectedStep:n,customStepTypes:a,customVariables:i,onButtonClick:o,onStepCompletion:s,flowId:l,type:c,hideOnFlowCompletion:d,onComplete:u,setVisible:p,setShowModal:h,onDismiss:f,showPagination:g=!1,customFormElements:y,allowBackNavigation:v,validationHandler:b,onFormDataChange:x,showFooter:w,prefillData:_,updateUrlOnPageChange:k,repeatable:S})=>{var E;let A=j(j({},{linkCollection:Jt,multiInput:Ct,callToAction:un,selectList:xn}),a),{primaryCTAClickSideEffects:P,secondaryCTAClickSideEffects:N}=Ht(),[C,O]=r.useState(!1),[I,M]=r.useState({}),[D,L]=r.useState(!1),[$,R]=r.useState(null),F=null!=(E=t[n])?E:null,{markStepCompleted:U,markStepStarted:B,isLoading:z,updateCustomVariables:V,markFlowCompleted:H}=oe();function Y(){var e;return{data:null!=(e=I[t[n].id])?e:{},stepId:t[n].id,customVariables:i}}function W(e,r,a){let i=n+1<t.length?t[n+1]:null;return s&&s(e,a,i,I,Y()),!o||o(e,n,r,i)}r.useEffect((()=>{V(i)}),[i,z]),r.useEffect((()=>{x&&x(I,Y(),t[n],n)}),[I]);let q=w&&m.default.createElement(Bt,{step:t[n],canContinue:C&&!D,formType:c,selectedStep:n,appearance:e,onPrimaryClick:()=>T(void 0,null,(function*(){if(L(!0),b){let e=yield b(t[n],n,t[n+1],I,Y());if(!0!==e)return R("string"==typeof e?e:null),void L(!1);R(null)}let e=j({},Y());yield U(l,t[n].id,e),n+1<t.length&&!v&&(yield B(l,t[n+1].id));let r=W(t[n],"primary",n);if(n+1>=t.length&&(u&&u(),f&&f(),d&&r&&(p&&p(!1),h(!1)),yield H(l),S&&(yield B(l,t[0].id),"undefined"!=typeof window&&window.localStorage&&Object.keys(window.localStorage).forEach((e=>{e.startsWith(Nt)&&window.localStorage.removeItem(e)})))),P(t[n]),L(!1),"undefined"!=typeof window&&!v&&k&&n+1<t.length){let e=new URL(window.location.href);e.searchParams.set("p",t[n+1].id),window.history.pushState({},"",e.toString())}})),onSecondaryClick:()=>{W(t[n],"secondary",n),N(t[n])},onBack:()=>T(void 0,null,(function*(){n-1>=0&&(L(!0),yield B(l,t[n-1].id),L(!1))})),steps:t,allowBackNavigation:v,errorMessage:$,isSaving:D});return m.default.createElement(m.default.Fragment,null,m.default.createElement(Mt,{className:ue("formContainer",e)},m.default.createElement(wn,{id:n,shouldWrap:"large-modal"===c},m.default.createElement(Dt,{key:F.id,type:c,className:ue("formContent",e)},t.map((t=>{var n;let r=A[null!=(n=t.type)?n:"multiInput"];return F.id!==t.id?null:"function"!=typeof r?r:m.default.createElement(r,{key:t.id,stepData:t,canContinue:C,setCanContinue:O,onSaveData:e=>{!function(e,t){M((n=>{let r={};return r[e.id]=t,j(j({},n),r)}))}(t,e)},appearance:e,customFormElements:y,flowId:l,prefillData:_})})),g&&m.default.createElement(Vt,{className:ue("formPagination",e),appearance:e,stepCount:t.length,currentStep:n}),q)),"large-modal"==c&&m.default.createElement((function(t){return t.selectedStep.imageUri?m.default.createElement(Lt,{image:t.selectedStep.imageUri,appearance:e,className:ue("formContainerSidebarImage",e)}):null}),{selectedStep:t[n]})))},kn=a.createGlobalStyle`
${e=>e.inlineStyles.map((([e,t])=>`.${de}${e}.${de}${e} { ${Object.entries(t).map((([e,t])=>"object"==typeof t?`${e} { ${Object.entries(t).map((([e,t])=>`${he(e)}: ${t};`)).join(" ")} }`:`${he(e)}: ${t};`)).join(" ")} }`)).join(" ")}`;function Sn({appearance:e}){if(!e||!e.styleOverrides)return m.default.createElement(m.default.Fragment,null);let t=Object.entries(e.styleOverrides).filter((([e,t])=>"object"==typeof t));return 0===t.length?m.default.createElement(m.default.Fragment,null):m.default.createElement(kn,{inlineStyles:t})}function En(e,t=!0){let[n,a]=r.useState(!1),{markStepStarted:i,isLoading:o,getFlowStatus:s,getFlowSteps:l,getCurrentStepIndex:c,targetingLogicShouldHideFlow:d,getFlow:u}=oe(),p=l(e);return r.useEffect((()=>{!function(){T(this,null,(function*(){!n&&!o&&s(e)===R&&!1===d(u(e))&&t&&p&&p.length>0&&(a(!0),yield i(e,p[c(e)].id))}))}()}),[o,e,t]),{}}var An=({flowId:e,customStepTypes:t={},type:n="inline",visible:a,setVisible:i,customVariables:o,customFormElements:s,onComplete:l,appearance:c,hideOnFlowCompletion:d=!0,onStepCompletion:u,onButtonClick:p,dismissible:h=!0,endFlowOnDismiss:f=!1,modalPosition:g="center",repeatable:y=!1,onDismiss:v,showPagination:b=!1,allowBackNavigation:x=!1,validationHandler:w,showFrigadeBranding:_=!1,onFormDataChange:k,showFooter:S=!0,prefillData:E={},updateUrlOnPageChange:A=!1})=>{let{getFlow:P,getFlowSteps:j,isLoading:N,targetingLogicShouldHideFlow:C,getFlowStatus:O,getCurrentStepIndex:T,markFlowCompleted:I,markFlowNotStarted:M,markStepStarted:L}=oe(),$=T(e),{mergeAppearanceWithDefault:R}=Ye(),[F,U]=r.useState(null),{setOpenFlowState:B,getOpenFlowState:z,hasOpenModals:V}=K();En(e,a);let H=j(e);c=R(c);let[Y,W]=void 0!==a&&void 0!==i?[a,i]:[z(e,!0),t=>B(e,t)],q="undefined"!=typeof window?window.location.hash:null;if(r.useEffect((()=>{var t;if(H&&H.length>0&&x){let n="undefined"!=typeof window&&null!=(t=null==window?void 0:window.location)&&t.hash?window.location.hash.replace("#",""):"";if(H&&(null==H?void 0:H.length)>0){let t=-1;if(n){let e=n;t=H.findIndex((t=>t.id===e)),F===n&&(t=-1)}-1!==t&&(U(n),L(e,H[t].id))}}}),[q]),r.useEffect((()=>{!N&&H&&H.length&&"undefined"!=typeof window&&x&&(U(H[$].id),window.location.hash=H[$].id)}),[N,$,H]),N)return null;let G=P(e);if(!G||C(G)||!H||void 0!==a&&!1===a||O(e)===D&&d&&!y||("modal"==n||"corner-modal"==n)&&V(e))return null;let Z=()=>{W(!1),v&&v(),!0===f&&I(e)};if("center"==g&&"modal"===n||"large-modal"===n){let r={padding:"24px"};return"large-modal"===n?(r.width="85%",r.height="90%",r.maxHeight="800px",r.minHeight="500px",r.padding="0"):r.width="400px",m.default.createElement(Fe,{appearance:c,onClose:Z,visible:Y,style:r,dismissible:h,showFrigadeBranding:_},m.default.createElement(Sn,{appearance:c}),m.default.createElement(_n,{appearance:c,steps:H,selectedStep:$,customStepTypes:t,customVariables:o,onButtonClick:p,onStepCompletion:u,flowId:e,type:n,hideOnFlowCompletion:d,onComplete:l,setVisible:i,setShowModal:W,onDismiss:v,showPagination:b,customFormElements:s,allowBackNavigation:x,validationHandler:w,onFormDataChange:k,showFooter:S,prefillData:E,updateUrlOnPageChange:A,repeatable:y}))}return"modal"===n&&"center"!==g?m.default.createElement(He,{appearance:c,onClose:Z,visible:Y,modalPosition:g},m.default.createElement(Sn,{appearance:c}),m.default.createElement(_n,{appearance:c,steps:H,selectedStep:$,customStepTypes:t,customVariables:o,onButtonClick:p,onStepCompletion:u,flowId:e,type:n,hideOnFlowCompletion:d,onComplete:l,setVisible:i,setShowModal:W,onDismiss:v,showPagination:b,customFormElements:s,allowBackNavigation:x,validationHandler:w,onFormDataChange:k,showFooter:S,prefillData:E,updateUrlOnPageChange:A,repeatable:y})):m.default.createElement(m.default.Fragment,null,m.default.createElement(Sn,{appearance:c}),m.default.createElement(_n,{appearance:c,steps:H,selectedStep:$,customStepTypes:t,customVariables:o,onButtonClick:p,onStepCompletion:u,flowId:e,type:n,hideOnFlowCompletion:d,onComplete:l,setVisible:i,setShowModal:W,onDismiss:v,showPagination:b,customFormElements:s,allowBackNavigation:x,validationHandler:w,onFormDataChange:k,showFooter:S,prefillData:E,updateUrlOnPageChange:A,repeatable:y}))},Pn=An;function jn(){let{organizationId:e,userId:t,setOrganizationId:n}=r.useContext(Yn),{mutateUserFlowState:a}=Q(),{config:i,apiUrl:o}=B(),s=H(),{verifySDKInitiated:l}=W();function c(e,t){return`frigade-user-group-registered-${e}-${t}`}r.useEffect((()=>{if(t&&e){if(t.startsWith(se))return;let n=c(t,e);localStorage.getItem(n)||(s(`${o}userGroups`,N(j({},i),{method:"POST",body:JSON.stringify({foreignUserId:t,foreignUserGroupId:e})})),localStorage.setItem(n,"true"))}}),[t,e]);let d=r.useCallback((n=>T(this,null,(function*(){if(!l())return;if(!e||!t)return;let r={foreignUserId:t,foreignUserGroupId:e,properties:n};yield s(`${o}userGroups`,N(j({},i),{method:"POST",body:JSON.stringify(r)})),a()}))),[e,t,i,a]),u=r.useCallback(((n,r)=>T(this,null,(function*(){if(!l())return;if(!e||!t)return;let c={foreignUserId:t,foreignUserGroupId:e,events:[{event:n,properties:r}]};yield s(`${o}userGroups`,N(j({},i),{method:"POST",body:JSON.stringify(c)})),a()}))),[e,t,i,a]),p=r.useCallback(((e,r)=>T(this,null,(function*(){if(l())if(r){let l=c(t,e);localStorage.setItem(l,"true"),n(e);let d={foreignUserId:t,foreignUserGroupId:e,properties:r};yield s(`${o}userGroups`,N(j({},i),{method:"POST",body:JSON.stringify(d)})),a()}else n(e)}))),[t,i,a]);return{organizationId:e,setOrganizationId:n,setOrganizationIdWithProperties:p,addPropertiesToOrganization:d,trackEventForOrganization:u}}var Nn="frigade-xFrigade_guestUserId",Cn="frigade-xFrigade_userId",On=({})=>{let{setFlowResponses:e}=ee(),{userFlowStatesData:t,isLoadingUserFlowStateData:n,mutateUserFlowState:a}=Q(),{userId:i,setUserId:o}=le(),[s,c]=r.useState(i),{getFlowStatus:d}=oe(),{flows:u,userProperties:p,setIsNewGuestUser:h,flowResponses:f,disableImagePreloading:g}=r.useContext(Yn),[y,v]=r.useState([]),[b,x]=r.useState([]),{organizationId:w}=jn(),[_,k]=r.useState(w),[S,E]=r.useState(!1);function A(e){let t=u.find((t=>t.slug===e));t&&"AUTOMATIC"===t.triggerType&&!b.includes(t.slug)&&(x([...b,t.slug]),v([t]))}return r.useEffect((()=>{if(!n&&t)for(let e=0;e<t.length;e++){let n=t[e],r=u.find((e=>e.slug===(null==n?void 0:n.flowId)));if(r&&n&&!0===n.shouldTrigger&&"FORM"==r.type&&"AUTOMATIC"===r.triggerType&&!b.includes(r.slug)){setTimeout((()=>{A(n.flowId)}),500);break}}}),[n,t]),r.useEffect((()=>{f.length>0&&a()}),[f]),r.useEffect((()=>{S||(E(!0),a())}),[n,E]),r.useEffect((()=>{try{if(!g&&u){let e=[];u.forEach((t=>{if(t.data&&t.active){let n=t.data.match(/"imageUri":"(.*?)"/g);n&&n.forEach((t=>{let n=t.replace('"imageUri":"',"").replace('"',"");e.includes(n)||((new Image).src=n,e.push(n))}))}}))}}catch(e){}}),[u]),r.useEffect((()=>{if(i!==s&&(e([]),a()),c(i),i&&!i.startsWith(se))try{localStorage.setItem(Cn,i)}catch(e){}null===i&&setTimeout((()=>{null===i&&function(){if(!i){let e=localStorage.getItem(Cn);if(e)return void o(e);let t=localStorage.getItem(Nn);if(t)return void o(t);h(!0);let n=se+l.v4();try{localStorage.setItem(Nn,n)}catch(e){}o((e=>e||n))}}()}),50)}),[i,u,p]),r.useEffect((()=>{w!=_&&(k(w),e([]),a())}),[w,_,k]),m.default.createElement(m.default.Fragment,null,m.default.createElement((function(){return m.default.createElement(m.default.Fragment,null,y.map((e=>d(e.slug)!==R?null:m.default.createElement("span",{key:e.slug},m.default.createElement(Pn,{flowId:e.slug,type:"modal",modalPosition:"center",endFlowOnDismiss:!0})))))}),null))},Tn={colorPrimary:"colors.primary.background",colorText:"colors.neutral.foreground",colorBackground:"colors.neutral.background",colorBackgroundSecondary:"colors.secondary.background",colorTextOnPrimaryBackground:"colors.primary.foreground",colorTextSecondary:"colors.secondary.foreground",colorTextDisabled:"colors.gray700",colorBorder:"colors.gray800",colorTextError:"colors.negative.foreground",borderRadius:"radii.lg"};function In(e){let{theme:t,styleOverrides:n}=e,r=function(e){if(!e)return;let t={};return Object.entries(e).forEach((([e,n])=>{if(Tn[e]){let r=Tn[e].split("."),a=t;r.forEach(((e,t)=>{a[e]||(a[e]=t===r.length-1?n:{}),a=a[e]}))}})),t}(t),a=function(e){if(!e)return;let t=Z({},e),n={};return Object.keys(t).forEach((e=>{n[`.fr-${e}`]=t[e]})),n}(n);return{overrides:r,css:a}}var Mn={width:{property:"width",scale:"sizes",transform:(e,t)=>p.get(t,e,"number"!=typeof e||isNaN(e)||e>1?e:100*e+"%")},height:{property:"height",scale:"sizes"},minWidth:{property:"minWidth",scale:"sizes"},minHeight:{property:"minHeight",scale:"sizes"},maxWidth:{property:"maxWidth",scale:"sizes"},maxHeight:{property:"maxHeight",scale:"sizes"},overflow:!0,overflowX:!0,overflowY:!0,display:!0,verticalAlign:!0},Dn=g.default("div")((({css:e})=>e),p.compose(p.border,p.color,p.shadow,p.space,p.typography,p.system(Mn))),Ln=e=>{var t=e,{as:n,children:r,overrides:i}=t,o=O(t,["as","children","overrides"]);let s=a.useTheme(),l={border:"none",boxSizing:"border-box",m:0,p:0},c=()=>m.default.createElement(Dn,j(j({as:n},l),o),r);if(void 0!==i){let e=Z(s,i);return m.default.createElement(a.ThemeProvider,{theme:e},c())}return c()},$n={Primary:{backgroundColor:"primary.background",color:"primary.foreground","&:hover":{backgroundColor:"blue400"}},Secondary:{backgroundColor:"white",border:"1px solid",borderColor:"gray800",color:"neutral.foreground","&:hover":{backgroundColor:"blue900"}},Link:{backgroundColor:"transparent",color:"primary.inverted"},Plain:{backgroundColor:"transparent",color:"neutral.foreground"}},Rn=g.default(Ln)((()=>({whiteSpace:"nowrap"})),p.compose(p.variant({scale:"components.Button",variants:"components.Button"}),p.variant({prop:"size",variants:{sm:{paddingX:4,paddingY:1},md:{paddingX:6,paddingY:2}}}))),Fn={Display1:{fontSize:"5xl",fontWeight:"bold",letterSpacing:"md",lineHeight:"4xl"},Display2:{fontSize:"4xl",fontWeight:"bold",letterSpacing:"md",lineHeight:"3xl"},H1:{fontSize:"3xl",fontWeight:"bold",letterSpacing:"md",lineHeight:"2xl"},H2:{fontSize:"2xl",fontWeight:"bold",letterSpacing:"md",lineHeight:"xl"},H3:{fontSize:"xl",fontWeight:"bold",letterSpacing:"md",lineHeight:"lg"},H4:{fontSize:"lg",fontWeight:"bold",letterSpacing:"md",lineHeight:"md"},Body1:{fontSize:"md",fontWeight:"regular",letterSpacing:"md",lineHeight:"md"},Body2:{fontSize:"sm",fontWeight:"regular",letterSpacing:"md",lineHeight:"md"},Caption:{fontSize:"xs",fontWeight:"regular",letterSpacing:"md",lineHeight:"sm"}},Un=g.default(Ln)(p.variant({scale:"components.Text",variants:"components.Text"}),p.system({fontWeight:{property:"fontWeight",scale:"fontWeights"}})),Bn=Object.fromEntries(Array.from(Array(21),((e,t)=>0===t?[.5,"2px"]:[t,4*t+"px"]))),zn={black:"#000000",gray100:"#14161A",gray200:"#181B20",gray300:"#1F2329",gray400:"#2E343D",gray500:"#4C5766",gray600:"#5A6472",gray700:"#C5CBD3",gray800:"#E2E5E9",gray900:"#F1F2F4",white:"#ffffff",blue400:"#015AC6",blue500:"#0171F8",blue800:"#DBECFF",blue900:"#F5F9FF",green400:"#009E37",green500:"#00D149",green800:"#DBFFE8",transparent:"#FFFFFF00",red500:"#c00000"},Vn={colors:N(j({},zn),{neutral:{foreground:zn.gray300},primary:{background:zn.blue500,foreground:zn.white,inverted:zn.blue500},negative:{foreground:zn.red500}}),fonts:{default:"TT Interphases Pro, sans-serif"},fontSizes:{xs:"12px",sm:"14px",md:"16px",lg:"18px",xl:"20px","2xl":"24px","3xl":"30px","4xl":"36px","5xl":"48px"},fontWeights:{regular:400,semibold:500,bold:700},letterSpacings:{md:"0.02em"},lineHeights:{xs:"18px",sm:"22px",md:"24px",lg:"26px",xl:"30px","2xl":"38px","3xl":"46px","4xl":"60px"},radii:{md:"8px",lg:"20px",round:"50%"},shadows:{md:"0px 4px 20px rgba(0, 0, 0, 0.06)"},space:Bn,components:{Button:$n,Text:Fn}},Hn="https://api.frigade.com",Yn=r.createContext({publicApiKey:"",setUserId:()=>{},flows:[],setFlows:()=>{},failedFlowResponses:[],setFailedFlowResponses:()=>{},flowResponses:[],setFlowResponses:()=>{},userProperties:{},setUserProperties:()=>{},openFlowStates:{},setOpenFlowStates:()=>{},completedFlowsToKeepOpenDuringSession:[],setCompletedFlowsToKeepOpenDuringSession:()=>{},customVariables:{},setCustomVariables:()=>{},isNewGuestUser:!1,setIsNewGuestUser:()=>{},hasActiveFullPageFlow:!1,setHasActiveFullPageFlow:()=>{},organizationId:"",setOrganizationId:()=>{},navigate:()=>{},defaultAppearance:Ze,shouldGracefullyDegrade:!1,setShouldGracefullyDegrade:()=>{},apiUrl:Hn,readonly:!1,debug:!1,disableImagePreloading:!1});var Wn=g.default.span`
  font-weight: 600;
  font-size: 14px;
  font-style: normal;
  line-height: 22px;
  letter-spacing: 0.28px;
  color: #4d4d4d;
  display: inline-block;
  vertical-align: middle;
  margin-left: 32px;
  padding-right: 12px;
  ${e=>fe(e)}
`,qn=g.default.div`
  flex-direction: row;
  justify-content: space-between;
  display: flex;
  padding-top: 20px;
  padding-bottom: 20px;
  padding-right: 8px;
  width: 100%;
  ${e=>fe(e)}
`,Kn=({label:e,value:t,labelStyle:n={},labelPosition:r="right",style:a,primaryColor:i="#000000",checkBoxType:o="round",appearance:s})=>m.default.createElement(m.default.Fragment,null,m.default.createElement(qn,{className:ue("checklistStepsContainer",s),appearance:s,styleOverrides:j({},a)},"left"===r&&e&&m.default.createElement(Wn,{className:ue("checklistStepLabel",s),styleOverrides:n,appearnace:s,dangerouslySetInnerHTML:yt(e)}),m.default.createElement(gt,{appearance:s,value:t,type:o,primaryColor:i}),"right"===r&&e&&m.default.createElement(Wn,{className:ue("checklistStepLabel",s),styleOverrides:n,appearance:s,dangerouslySetInnerHTML:yt(e)})),m.default.createElement(Sn,{appearance:s})),Gn=g.default.div`
  font-weight: 700;
  font-size: 18px;
  line-height: 22px;
`,Zn=g.default.p`
  font-weight: 600;
  font-size: 16px;
  line-height: 24px;
  margin: 20px 0px 0px 0px;
  letter-spacing: 0.32px;
  font-style: normal;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
`,Jn=g.default.p`
  font-weight: 400;
  font-size: 14px;
  font-style: normal;
  line-height: 22px;
  max-width: 540px;
  letter-spacing: 0.28px;
  margin: 8px 0px 0px 0px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
`,Xn=g.default.div`
  width: 6px;
  position: absolute;
  left: 0;
  height: 100%;
  border-top-right-radius: 4px;
  border-bottom-right-radius: 4px;
`,Qn=g.default.div`
  flex-direction: row;
  justify-content: flex-start;
  border-bottom: 1px solid ${e=>e.theme.colorBorder};
  padding-right: 16px;
`,er=({data:e,index:t,isSelected:n,primaryColor:r,style:a,onClick:i,appearance:o})=>{var s,l,c,d,u;return m.default.createElement("div",{style:{position:"relative",paddingLeft:"0px"},onClick:()=>{i()}},n&&m.default.createElement(Xn,{className:ue("checklistStepItemSelectedIndicator",o),layoutId:"checklis-step-selected",style:{backgroundColor:null!=(l=null==(s=null==o?void 0:o.theme)?void 0:s.colorPrimary)?l:r}}),m.default.createElement(Qn,{className:ue("checklistStepItem",o),key:`hero-checklist-step-${t}`,appearance:o,role:"listitem"},m.default.createElement(Kn,{value:e.complete,labelPosition:"left",label:null!=(c=e.stepName)?c:e.title,style:a,primaryColor:null!=(u=null==(d=null==o?void 0:o.theme)?void 0:d.colorPrimary)?u:r,appearance:o})))};g.default.h1`
  display: flex;
  margin: 0;
  font-size: 18px;
`,g.default.h2`
  font-size: 15px;
  color: #4d4d4d;
`;var tr=g.default.div`
  position: absolute;
  left: 0;
  top: 0;
  height: ${e=>e.barHeight};
  width: ${e=>e.fgWidth};
  border-radius: 20px;
  background-color: ${e=>e.theme.colorPrimary};
  transition: width 0.5s;
`,nr=g.default.div`
  position: relative;
  left: 0;
  top: 0;
  width: 100%;
  min-width: 40px;
  height: ${e=>e.barHeight};
  border-radius: 20px;
  background-color: ${e=>e.theme.colorPrimary};
  opacity: 0.12;
`,rr=g.default.div`
  display: flex;
  flex-direction: ${e=>"top"==e.textLocation?"column":"row"};
  justify-content: flex-start;
  align-items: ${e=>"top"==e.textLocation?"flex-end":"center"};
  width: 100%;

  ${e=>fe(e)}
`,ar=g.default.div`
  flex-grow: 1;
  position: relative;
  ${e=>"top"==e.textLocation?"width: 100%;":""}
`,ir=g.default.span`
  font-weight: 600;
  font-size: 14px;
  line-height: 18px;
  padding-right: ${e=>e.padding};
  margin-bottom: ${e=>"top"==e.textLocation?"8px":"0px"};
  ${e=>fe(e)}
`,or=({count:e,total:t,display:n="count",textLocation:r="left",style:a={},textStyle:i={},appearance:o})=>{var s;if(0===t)return m.default.createElement(m.default.Fragment,null);0===Object.keys(i).length&&i.constructor===Object&&(i={color:null==(s=null==o?void 0:o.theme)?void 0:s.colorText});let l,c=0===e?"10px":e/t*100+"%",d="compact"===n?"8px":"10px",u=Math.round(e/t*100),p="compact"===n?"10px":"14px";return"count"===n||"compact"===n?l=`${e}/${t}`:"percent"===n&&(l=`${u}% complete`),"top"===r&&(p="0px"),m.default.createElement(rr,{className:ue("progressBarContainer",o),textLocation:r,styleOverrides:a},m.default.createElement(ir,{className:ue("progressBarStepText",o),styleOverrides:N(j({},i),{fontSize:"compact"===n?12:14,fontWeight:600}),appearance:o,padding:p,textLocation:r},l),m.default.createElement(ar,{textLocation:r,className:ue("progressBar",o)},m.default.createElement(tr,{style:{zIndex:"compact"==n?1:5},fgWidth:c,barHeight:d,appearance:o,className:ue("progressBarFill",o)}),m.default.createElement(nr,{className:ue("progressBarBackground",o),barHeight:d,appearance:o})))},sr=({stepData:e,appearance:t})=>m.default.createElement(m.default.Fragment,null,m.default.createElement(Zn,{appearance:t,className:ue("checklistStepTitle",t),dangerouslySetInnerHTML:yt(e.title)}),m.default.createElement(Jn,{appearance:t,className:ue("checklistStepSubtitle",t),dangerouslySetInnerHTML:yt(e.subtitle)})),lr=({stepData:e,appearance:t})=>m.default.createElement(Ft,{className:ue("ctaContainer",t)},m.default.createElement(Ut,{appearance:t,title:e.primaryButtonTitle,onClick:()=>{e.handlePrimaryButtonClick&&e.handlePrimaryButtonClick()}}),e.secondaryButtonTitle&&m.default.createElement(Ut,{appearance:t,secondary:!0,title:e.secondaryButtonTitle,onClick:()=>{e.handleSecondaryButtonClick&&e.handleSecondaryButtonClick()},style:{width:"auto",marginRight:"12px"}})),cr=({stepData:e,appearance:t})=>m.default.createElement(m.default.Fragment,null,m.default.createElement(sr,{stepData:e,appearance:t}),m.default.createElement(lr,{stepData:e,appearance:t}));function dr(e){return m.default.createElement(on,{appearance:e.appearance,videoUri:e.videoUri,autoplay:e.autoplay,loop:e.loop,hideControls:e.hideControls})}var ur="default",pr=g.default.img`
  border-radius: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.borderRadius}}px;
  width: 100%;
  height: auto;
  min-height: 200px;
`,hr=({stepData:e,appearance:t})=>{var n,r,a;if(null!=e&&e.StepContent){let t=e.StepContent;return m.default.createElement("div",null,t)}return m.default.createElement(Gn,{className:ue("checklistStepContent",t)},e.imageUri?m.default.createElement(pr,{className:ue("checklistStepImage",t),src:e.imageUri,appearance:t}):null,e.videoUri?m.default.createElement(dr,{videoUri:e.videoUri,appearance:t,autoplay:null==(n=e.props)?void 0:n.autoplayVideo,loop:null==(r=e.props)?void 0:r.loopVideo,hideControls:null==(a=e.props)?void 0:a.hideVideoControls}):null,m.default.createElement(cr,{stepData:e,appearance:t}))},fr=g.default.div`
  display: block;
`,mr=g.default.div`
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-start;
  gap: 0px;
  align-items: center;
  align-content: center;
  margin-top: 24px;
  margin-bottom: 24px;
`,gr=g.default.div`
  display: flex;
  align-items: center;
  justify-content: flex-start;
  flex-direction: column;
  margin-right: 16px;
`,yr=g.default.video`
  width: 200px;
  height: 120px;
`,vr=g.default.div`
  font-size: 14px;
  line-height: 20px;
  text-align: center;
`,br=g.default.div`
  position: absolute;
  width: 200px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  :hover {
    opacity: 0.6;
  }
  z-index: 10;

  > svg {
    width: 40px;
    height: 40px;
    color: ${e=>e.appearance.theme.colorBackground};
    box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.06);
    border-radius: 50%;
  }
`,xr=({stepData:e,appearance:t})=>{var n;if(null==(n=e.props)||!n.videos)return m.default.createElement(fr,null,m.default.createElement(cr,{stepData:e,appearance:t}));function a({video:e}){let n=r.useRef(),[a,i]=r.useState(!1);return m.default.createElement(gr,null,!a&&m.default.createElement(br,{onClick:()=>{i(!0),n.current.play()},appearance:t},m.default.createElement(Xt,null)),m.default.createElement(yr,{controls:a,ref:n,play:a,src:e.uri}),m.default.createElement(vr,null,e.title))}let i=e.props;return i.videos?m.default.createElement(fr,null,m.default.createElement(sr,{stepData:e,appearance:t}),m.default.createElement(mr,null,i.videos.map(((e,t)=>m.default.createElement("span",{key:`${e.uri}-${t}`},m.default.createElement(a,{video:e}))))),m.default.createElement(lr,{stepData:e,appearance:t})):null},wr=g.default.div`
  display: block;
`,_r=g.default.pre`
  display: block;
  background-color: #2a2a2a;
  color: #f8f8f8;
  padding: 16px;
  border-radius: 6px;
  font-size: 14px;
  line-height: 20px;
  font-family: 'Source Code Pro', monospace;
  width: 600px;
  white-space: pre-wrap; /* css-3 */
  white-space: -moz-pre-wrap; /* Mozilla, since 1999 */
  white-space: -pre-wrap; /* Opera 4-6 */
  white-space: -o-pre-wrap; /* Opera 7 */
  word-wrap: break-word; /* Internet Explorer 5.5+ */
  margin-bottom: 24px;
`,kr=g.default.div`
  font-size: 15px;
  line-height: 20px;
  margin-bottom: 12px;
  margin-top: 12px;
`,Sr=g.default.div`
  margin-top: 24px;
`,Er=({stepData:e,appearance:t})=>{var n;if(null==(n=e.props)||!n.codeSnippets)return m.default.createElement(wr,null,m.default.createElement(cr,{stepData:e,appearance:t}));let r=e.props;return r.codeSnippets?m.default.createElement(wr,{className:ue("codeSnippetContainer",t)},m.default.createElement(sr,{stepData:e,appearance:t}),m.default.createElement(Sr,null,r.codeSnippets.map(((e,t)=>m.default.createElement("div",{key:t},e.title?m.default.createElement(kr,{dangerouslySetInnerHTML:yt(e.title)}):null,e.code?m.default.createElement(_r,null,e.code):null)))),m.default.createElement(lr,{stepData:e,appearance:t})):null},Ar=g.default.div`
  display: flex;
  flex-direction: row;
  overflow: hidden;
  min-width: ${e=>"modal"!=e.type?"600px":"100%"};
  background: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.colorBackground}};
  border-radius: ${e=>{var t;return null==(t=e.appearance)?void 0:t.theme.borderRadius}}px;
  ${e=>fe(e)}
`,Pr=g.default.h1`
  font-size: 18px;
  font-style: normal;
  font-weight: 700;
  line-height: 24px;
  letter-spacing: 0.36px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
`,jr=g.default.h2`
  font-size: 14px;
  font-style: normal;
  font-weight: 400;
  line-height: 22px;
  letter-spacing: 0.28px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
  margin: 10px 0px 0px 0px;
`,Nr=g.default.div`
  padding: 28px 0px 28px 28px;
  border-bottom: 1px solid ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
`,Cr=g.default.div`
  flex: 1;
`,Or=g.default.div`
  list-style: none;
  padding: 0;
  margin: 0;
  cursor: pointer;
  min-width: 300px;
`,Tr=g.default.div`
  width: 1px;
  margin-right: 40px;
  background: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
`,Ir=g.default.div`
  flex: 2;
  padding: 40px 40px 40px 0px;
`,Mr=({title:e,subtitle:t,steps:n=[],style:a={},selectedStep:i,setSelectedStep:o,className:s="",customStepTypes:l=new Map,appearance:c,type:d})=>{let{mergeAppearanceWithDefault:u}=Ye();c=u(c);let p=j(j({},{[ur]:hr,videoCarousel:xr,codeSnippet:Er}),l),[h,f]=r.useState(0),g=null!=i?i:h,y=null!=o?o:f,v=n.filter((e=>!0===e.complete)).length;return m.default.createElement(Ar,{type:d,styleOverrides:a,className:s,appearance:c},m.default.createElement(Cr,{className:ue("checklistHeaderContainer",c),appearance:c},m.default.createElement(Nr,{className:ue("checklistHeader",c),appearance:c},m.default.createElement(Pr,{className:ue("checklistTitle",c),appearance:c,dangerouslySetInnerHTML:yt(e)}),m.default.createElement(jr,{className:ue("checklistSubtitle",c),appearance:c,dangerouslySetInnerHTML:yt(t)}),m.default.createElement(or,{total:n.length,count:v,style:{marginTop:"24px",paddingRight:"40px"},appearance:c})),m.default.createElement(Or,{className:ue("checklistStepsContainer",c)},n.map(((e,t)=>m.default.createElement(er,{data:e,index:t,key:t,listLength:n.length,isSelected:t===g,primaryColor:c.theme.colorPrimary,style:{justifyContent:"space-between"},onClick:()=>{y(t)},appearance:c}))))),m.default.createElement(Tr,{appearance:c,className:ue("checklistDivider",c)}),m.default.createElement(Ir,null,m.default.createElement((()=>{var e;return null!=(e=n[g])&&e.type&&p[n[g].type]?"function"!=typeof p[n[g].type]?p[n[g].type]:p[n[g].type]({stepData:n[g],appearance:c}):p[ur]({stepData:n[g],appearance:c})}),null)))},Dr=g.default.svg`
  transition: 'transform 0.35s ease-in-out';
`,Lr=({style:e,className:t})=>m.default.createElement(Dr,{width:"7",height:"10",viewBox:"0 0 9 15",fill:"none",xmlns:"http://www.w3.org/2000/svg",style:e,className:t},m.default.createElement("path",{d:"M1 13L7.5 7L0.999999 1",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round",strokeLinejoin:"round"})),$r={boxShadow:"0px 6px 25px rgba(0, 0, 0, 0.06)",padding:"32px",maxHeight:"700px",msOverflowStyle:"none",scrollbarWidth:"none",paddingBottom:"12px",minHeight:"610px"},Rr=g.default.div`
  max-height: 350px;
  padding-bottom: 40px;
`,Fr=g.default.div`
  display: flex;
  flex-direction: column;
  margin-top: 20px;
`,Ur=g.default.h1`
  font-style: normal;
  font-weight: 600;
  font-size: 30px;
  line-height: 36px;
  margin-bottom: 16px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
`,Br=g.default.h2`
  font-weight: 400;
  font-size: 16px;
  line-height: 20px;
  margin-bottom: 16px;
  padding-left: 1px;
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
`,zr=g.default.div`
  ${e=>pe(e)} {
    border: 1px solid #fafafa;
  }
  box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
  border-radius: 14px;
  display: flex;
  flex-direction: column;
  min-height: 240px;
  overflow: hidden;
`,Vr=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
`,Hr=g.default.p`
  ${e=>pe(e)} {
    font-weight: 400;
    font-size: 10px;
    line-height: 12px;
    text-transform: uppercase;
    color: #8c8c8c;
    margin: 20px;
  }
`,Yr=g.default.div`
  display: flex;
  flex-direction: row;
`,Wr=g.default.div`
  flex: 1;
`,qr=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    justify-content: center;
    align-content: center;
    flex-direction: column;
    flex: 1;
    padding-left: 8px;
    padding-right: 8px;
  }
`,Kr=g.default.p`
  ${e=>pe(e)} {
    font-style: normal;
    font-weight: 600;
    font-size: 22px;
    line-height: 26px;

    text-align: center;
    color: ${e=>e.appearance.theme.colorText};
    margin-top: 20px;
    margin-bottom: 16px;
  }
`,Gr=g.default.p`
  ${e=>pe(e)} {
    font-weight: 400;
    font-size: 14px;
    line-height: 18px;
    text-align: center;
    color: ${e=>e.appearance.theme.colorTextSecondary};
    margin-bottom: 8px;
  }
`,Zr=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 8px;
  }
`,Jr=g.default.div`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background-color: ${e=>e.selected?"#FAFAFA":"#FFFFFF"};
    :hover {
      background-color: #fafafa;
    }
  }
  //Check if attr disabled is true
  &[disabled] {
    opacity: 0.3;
    cursor: not-allowed;
  }

  padding: 20px;
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  position: relative;
  cursor: pointer;
`,Xr=g.default.p`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    color: ${e=>e.selected?"#434343":"#BFBFBF"};
  }
  font-weight: ${e=>e.selected?500:400};
  font-size: 14px;
  line-height: 22px;
  margin: 0;
`,Qr=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-content: center;
`,ea=g.default.div`
  display: flex;
  flex: 1;
  flex-direction: row;
  justify-content: flex-end;
  align-content: center;
  align-items: center;
  margin-right: 20px;
`,ta=g.default.div`
  display: block;
  width: 100%;
`,na=g.default.div`
  flex-direction: column;
  justify-content: center;
  display: flex;
`,ra=g.default.div`
  border: 1px solid #fafafa;
  box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
  border-radius: 14px;
  padding-top: 20px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`,aa=g.default.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  overflow: hidden;
  row-gap: 10px;
`,ia=g.default.div`
  ${e=>pe(e)} {
    color: #595959;
  }
  text-transform: uppercase;
  font-weight: 400;
  font-size: 10px;
  line-height: 12px;
  letter-spacing: 0.09em;
  margin-bottom: 12px;
`,oa=g.default.div`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background: #ffffff;
    border: 1px solid #fafafa;
  }
  border-radius: 14px;
  padding: 20px;
  flex-direction: column;
  align-content: center;

  max-width: 150px;
  min-width: 200px;
`,sa=g.default.div`
  ${e=>pe(e)} {
    background: radial-gradient(50% 50% at 50% 50%, #ffffff 0%, #f7f7f7 100%);
  }
  width: 40px;
  height: 40px;

  border-radius: 7px;
  display: flex;
  justify-content: center;
  align-content: center;
  align-items: center;
`,la=g.default.div`
  font-weight: 600;
  font-size: 20px;
  line-height: 24px;
  width: 20px;
  height: 20px;
`,ca=g.default.div`
  ${e=>pe(e)} {
    color: #434343;
  }
  font-weight: 600;
  font-size: 14px;
  line-height: 17px;
  margin-top: 12px;
  margin-bottom: 8px;
`,da=g.default.div`
  ${e=>pe(e)} {
    color: #8c8c8c;
  }
  font-weight: 400;
  font-size: 12px;
  line-height: 14px;
`,ua=g.default.a`
  color: ${e=>e.color};
  font-size: 12px;
  line-height: 14px;
  font-weight: 400;
  cursor: pointer;
`,pa=({steps:e,style:t,title:n,primaryColor:r,appearance:a,onButtonClick:i})=>{let{primaryCTAClickSideEffects:o}=Ht();return m.default.createElement(ra,{style:t,className:ue("guideContainer",a)},m.default.createElement(ia,{className:ue("guideTitle",a)},n),m.default.createElement(aa,{className:ue("guideItemContainer",a)},e.map(((e,t)=>{var n;return m.default.createElement(oa,{key:`guide-${null!=(n=e.id)?n:t}`,className:ue("guideItem",a)},e.icon&&m.default.createElement(sa,{className:ue("guideIcon",a)},m.default.createElement(la,null,e.icon)),m.default.createElement(ca,{className:ue("guideItemTitle",a),dangerouslySetInnerHTML:yt(e.title)}),m.default.createElement(da,{className:ue("guideItemSubtitle",a),dangerouslySetInnerHTML:yt(e.subtitle)}),m.default.createElement(ua,{className:ue("guideItemLink",a),color:r,onClick:()=>{e.primaryButtonUri&&o(e),i&&i(e)}},e.primaryButtonTitle))}))))},ha=({steps:e,title:t,subtitle:n,stepsTitle:a,visible:i,onClose:o,selectedStep:s,setSelectedStep:l,customStepTypes:c,appearance:d,guideData:u,guideTitle:p,onGuideButtonClick:h})=>{let f=({stepData:e,handleSecondaryCTAClick:t,handleCTAClick:n})=>e?m.default.createElement(qr,{className:ue("checklistStepContainer",d),"data-testid":"checklistStepContainer"},m.default.createElement(Kr,{appearance:d,className:ue("checklistStepTitle",d),dangerouslySetInnerHTML:yt(e.title)}),m.default.createElement(Gr,{appearance:d,className:ue("checklistStepSubtitle",d),dangerouslySetInnerHTML:yt(e.subtitle)}),m.default.createElement(Zr,{className:ue("checklistCTAContainer",d)},e.secondaryButtonTitle&&m.default.createElement(Ut,{title:e.secondaryButtonTitle,onClick:t,appearance:d,secondary:!0}),m.default.createElement(Ut,{title:e.primaryButtonTitle,onClick:n,appearance:d}))):m.default.createElement(m.default.Fragment,null),g=j(j({},{default:t=>{var n;if(null!=(n=e[b])&&n.StepContent){let t=e[b].StepContent;return m.default.createElement("div",null,t)}let r=e[b];return m.default.createElement(f,{stepData:t,handleCTAClick:()=>{r.handlePrimaryButtonClick&&r.handlePrimaryButtonClick()},handleSecondaryCTAClick:()=>{r.handleSecondaryButtonClick&&r.handleSecondaryButtonClick()}})}}),c),[y,v]=r.useState(0),b=null!=s?s:y,x=null!=l?l:v,w=e.filter((e=>e.complete)).length;return i?(d.theme.modalContainer||(d.theme.borderRadius&&($r.borderRadius=d.theme.borderRadius+"px"),d.theme.modalContainer=$r),m.default.createElement(Fe,{onClose:o,visible:i,appearance:d},m.default.createElement(ta,null,m.default.createElement(Fr,null,m.default.createElement(Ur,{appearance:d,className:ue("checklistTitle",d)},t),m.default.createElement(Br,{appearance:d,className:ue("checklistSubtitle",d)},n)),m.default.createElement(Rr,null,e&&e.length>0&&m.default.createElement(zr,{className:ue("stepsContainer",d)},m.default.createElement(Vr,null,m.default.createElement("div",{style:{flex:3}},m.default.createElement(Hr,{className:ue("stepsTitle",d)},a)),m.default.createElement(ea,null,m.default.createElement(or,{style:{width:"100%"},count:w,total:e.length,appearance:d}))),m.default.createElement(Yr,null,m.default.createElement(Wr,{className:ue("checklistStepListContainer",d),appearance:d},e.map(((e,t)=>{var n;let r=b===t;return m.default.createElement(Jr,{selected:r,className:ue("checklistStepListItem"+(r?"Selected":""),d),key:`checklist-guide-step-${null!=(n=e.id)?n:t}`,disabled:e.blocked,onClick:()=>{e.blocked||x(t)},title:e.blocked?"Finish remaining steps to continue":void 0},r&&m.default.createElement(Xn,{className:ue("checklistStepItemSelectedIndicator",d),layoutId:"checklist-step-selected",style:{backgroundColor:d.theme.colorPrimary,borderRadius:0,height:"100%",top:"0%",width:"2px"}}),m.default.createElement(Xr,{selected:r,className:ue("checklistStepListStepName"+(r?"Selected":""),d)},e.stepName),m.default.createElement(Qr,null,m.default.createElement(gt,{value:e.complete,type:"round",primaryColor:d.theme.colorPrimary,progress:e.progress,appearance:d}),m.default.createElement(na,null,m.default.createElement(Lr,{style:{marginLeft:"10px"},color:d.theme.colorBackgroundSecondary}))))}))),m.default.createElement((()=>{var t;return e?null!=(t=e[b])&&t.type&&g[e[b].type]?"function"!=typeof g[e[b].type]?g[e[b].type]:g[e[b].type]({stepData:e[b],primaryColor:d.theme.colorPrimary}):g.default(e[b]):m.default.createElement(m.default.Fragment,null)}),null))),u&&u.length>0&&m.default.createElement(pa,{steps:u,title:p,primaryColor:d.theme.colorPrimary,style:{border:"none",boxShadow:"none"},appearance:d,onButtonClick:e=>(h&&h(e),!0)}))))):m.default.createElement(m.default.Fragment,null)},fa=g.default.div`
  background-color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBackground}};
  border: 1px solid;
  border-color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBorder}};
  border-radius: 6px;
  padding: 2px 20px 2px 20px;
  display: flex;
  margin-top: 14px;
  margin-bottom: 14px;
  display: flex;
  flex-direction: column;
  transition: max-height 0.25s;
`,ma=g.default.div`
  display: flex;
  margin-bottom: 20px;
`,ga=g.default.img`
  border-radius: 4px;
  max-height: 260px;
  min-height: 200px;
`,ya=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  :hover {
    opacity: 0.8;
  }
`,va=g.default.p`
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  font-style: normal;
  font-weight: 500;
  font-size: 16px;
  line-height: 18px;
  margin-left: 8px;
  cursor: pointer;
  :hover {
    opacity: 0.8;
  }
`,ba=g.default.div`
  cursor: pointer;
  color: ${e=>e.appearance.theme.colorTextSecondary};
  :hover {
    opacity: 0.8;
  }
`;g.default.div``;var xa=g.default.p`
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  font-weight: 400;
  font-size: 14px;
  line-height: 22px;
`,wa=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: flex-start;
  align-items: center;
`,_a=({stepData:e,collapsed:t,onClick:n,onPrimaryButtonClick:r,onSecondaryButtonClick:a,appearance:i,customStepTypes:o})=>{var s,l;let c=t?{}:{transform:"rotate(90deg)"},d=t?{overflow:"hidden",maxHeight:"0px",transition:"max-height 0.35s ease-out"}:{overflow:"hidden",maxHeight:"1000px",transition:"max-height 0.7s ease-out"};return m.default.createElement(fa,{"data-testid":`step-${e.id}`,className:ue("checklistStepContainer",i),appearance:i},m.default.createElement(ya,{className:ue("stepHeader",i),onClick:()=>n()},m.default.createElement(wa,{className:ue("stepHeaderContent",i)},m.default.createElement(Kn,{value:e.complete,style:{width:"auto",borderTop:0},primaryColor:null==(s=null==i?void 0:i.theme)?void 0:s.colorPrimary,appearance:i}),m.default.createElement(va,{appearance:i,className:ue("stepTitle",i),dangerouslySetInnerHTML:yt(e.title)})),m.default.createElement(ba,{className:ue("stepChevronContainer",i),appearance:i},m.default.createElement(Lr,{style:N(j({},c),{transition:"transform 0.2s ease-in-out"})}))),m.default.createElement("div",{key:e.id,style:j({},d),className:ue("stepContent",i)},null!=(l=function(){if(!o)return null;let t=o[e.type];return t?"function"!=typeof t?t:t(e,i):null}())?l:m.default.createElement(m.default.Fragment,null,e.imageUri||e.videoUri?m.default.createElement(ma,{className:ue("stepMediaContainer",i)},e.imageUri?m.default.createElement(ga,{className:ue("stepImage",i),src:e.imageUri}):null,e.videoUri?m.default.createElement(on,{appearance:i,videoUri:e.videoUri,autoplay:null==(u=e.props)?void 0:u.autoplayVideo,loop:null==(p=e.props)?void 0:p.loopVideo,hideControls:null==(h=e.props)?void 0:h.hideVideoControls}):null):null,m.default.createElement(xa,{className:ue("stepSubtitle",i),appearance:i,dangerouslySetInnerHTML:yt(e.subtitle)}),m.default.createElement(Ft,{className:ue("checklistCTAContainer",i)},e.secondaryButtonTitle?m.default.createElement(Ut,{secondary:!0,title:e.secondaryButtonTitle,onClick:()=>a(),appearance:i}):null,m.default.createElement(Ut,{title:null!=(f=e.primaryButtonTitle)?f:"Continue",onClick:()=>r(),appearance:i})))));var u,p,h,f};g.default.div`
  background: #ffffff;
  box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.06);
  border-radius: 6px;
  z-index: 10;
  padding: 32px;

  position: absolute;
  width: 80%;
  top: 20%;
  left: 20%;

  max-width: 800px;
  min-width: 350px;
`;var ka=g.default.div`
  display: flex;
  flex-direction: column;
`,Sa=g.default.h1`
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorText}};
  font-style: normal;
  font-weight: 700;
  font-size: 20px;
  line-height: 24px;
  margin-bottom: 8px;
`,Ea=g.default.h2`
  color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorTextSecondary}};
  font-weight: 400;
  font-size: 14px;
  line-height: 23px;
  margin: 2px 0 0 0;
`,Aa=g.default.div`
  display: block;
  width: 100%;
`,Pa=g.default.div`
  display: flex;
  width: 100%;
  flex-direction: column;
  justify-content: space-between;
  padding: 24px;
  border-radius: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.borderRadius}}px;
  background-color: ${e=>{var t,n;return null==(n=null==(t=e.appearance)?void 0:t.theme)?void 0:n.colorBackground}};
`,ja=({title:e,subtitle:t,steps:n,onClose:a,visible:i,autoExpandFirstIncompleteStep:o=!0,autoCollapse:s=!0,autoExpandNextStep:l=!0,setSelectedStep:c,appearance:d,type:u,className:p,customStepTypes:h,style:f,onButtonClick:g})=>{let y=n.filter((e=>e.complete)).length,[v,b]=r.useState(Array(n.length).fill(!0));r.useEffect((()=>{let e=[...v];if(o){for(let t=0;t<n.length;t++)if(!n[t].complete){e[t]=!1;break}b(e)}}),[]);let x=e=>{let t=[...v];if(s)for(let n=0;n<v.length;++n)n!=e&&t[e]&&(t[n]=!0);t[e]=!v[e],b(t)};if(!i&&"modal"==u)return m.default.createElement(m.default.Fragment,null);let w=m.default.createElement(m.default.Fragment,null,m.default.createElement(ka,null,m.default.createElement(Sa,{appearance:d,className:ue("checklistTitle",d),dangerouslySetInnerHTML:yt(e)}),m.default.createElement(Ea,{appearance:d,className:ue("checklistSubtitle",d),dangerouslySetInnerHTML:yt(t)})),m.default.createElement(or,{display:"percent",count:y,total:n.length,style:{margin:"14px 0px 8px 0px"},appearance:d})),_=m.default.createElement(Aa,{className:me(ue("checklistContainer",d),p)},n.map(((e,t)=>{var r;let a=v[t];return m.default.createElement(_a,{appearance:d,stepData:e,collapsed:a,key:`modal-checklist-${null!=(r=e.id)?r:t}`,onClick:()=>{x(t),c(t),g&&g(n[t],t,v[t]?"expand":"collapse")},onPrimaryButtonClick:()=>{x(t),l&&t<n.length-1&&c(t+1),e.handlePrimaryButtonClick&&e.handlePrimaryButtonClick()},onSecondaryButtonClick:()=>{e.handleSecondaryButtonClick&&e.handleSecondaryButtonClick()},customStepTypes:h})})));return"inline"===u?m.default.createElement(Pa,{appearance:d,className:me(ue("checklistInlineContainer",d),p),style:f},w,_):m.default.createElement(m.default.Fragment,null,m.default.createElement(Fe,{onClose:a,visible:i,appearance:d,style:{maxWidth:"600px"},headerContent:w},_))},Na=a.css`
  border: 1px solid ${({theme:e})=>e.colorBorder};
`,Ca=a.css`
  box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.06);
`,Oa=a.keyframes`
  from {
    opacity: 0;
  } to {
    opacity: 1;
  }
`,Ta=a.keyframes`
  from {
    opacity: 1;
  } to {
    opacity: 0;
  }
`,Ia=g.default.div`
  margin: 0 -20px;
  overflow-x: auto;
  padding-left: 20px;
  padding-right: 20px;
  scroll-snap-type: x mandatory;

  display: flex;
  flex-flow: row nowrap;
  gap: 0 16px;

  -ms-overflow-style: none;
  scrollbar-width: none;

  ::-webkit-scrollbar {
    display: none;
  }
`,Ma=g.default.div`
  display: flex;
  flex-flow: row nowrap;
  gap: 0 16px;
  scroll-snap-align: center;
  scroll-snap-stop: always;
`,Da=g.default.div`
  animation: ${e=>e.reversed?Ta:Oa} 0.25s ease-out;
  background: linear-gradient(
    to right,
    ${({theme:e})=>e.colorBackground}00,
    ${({theme:e})=>e.colorBackground} 100%
  );
  position: absolute;
  width: 64px;
  z-index: 10;
`,La=g.default.button`
  ${Na}
  box-shadow: 0 3px 10px 0 rgba(0, 0, 0, 0.1);
  align-items: center;
  border-radius: 50%;
  background: ${({theme:e})=>e.colorBackground};
  color: ${({theme:e})=>e.colorPrimary};
  display: flex;
  height: 48px;
  justify-content: center;
  position: absolute;
  width: 48px;
`,$a=g.default.div`
  border-radius: ${({theme:e})=>e.borderRadius}px;
  padding: 20px;
`,Ra=g.default($a)`
  ${Na}
  background: ${({theme:e})=>e.colorBackground};
  position: relative;

  &:active {
    ${e=>e.blocked?"":`background: ${e.theme.colorBackgroundSecondary};`}
  }

  &:hover {
    ${e=>e.blocked?"":`border: 1px solid ${e.theme.colorPrimary};`}
    ${e=>e.blocked?"cursor: default":"cursor: pointer"}
  }
`,Fa=g.default.img`
  border-radius: 50%;
  height: 40px;
  margin-bottom: 12px;
  width: 40px;
`,Ua=g.default($a)`
  ${e=>pe(e)} {
    ${Ca}

    background: ${({theme:e})=>e.colorBackground};
  }
`;g.default.div`
  color: ${({theme:e})=>e.colorPrimary};
  display: flex;
  flex-flow: row nowrap;
  align-items: center;
`,g.default.div`
  white-space: nowrap;
`;var Ba=g.default.div`
  background: #d8fed8;
  border-radius: 6px;
  float: right;
  margin-bottom: 12px;
  padding: 4px 10px;
`,za=g.default.p`
  font-weight: bold;
  font-size: 18px;
  line-height: 22px;
  letter-spacing: calc(18px * -0.01);
  margin: 0;
`,Va=g.default(za)`
  margin-bottom: 4px;
`,Ha=g.default.div`
  display: flex;
  flex-flow: row nowrap;
  align-items: center;
  min-width: 50%;
`,Ya=g.default.p`
  font-weight: bold;
  font-size: 16px;
  line-height: 20px;
  letter-spacing: calc(16px * -0.01);
  margin: 0;
`,Wa=g.default(Ya)`
  margin-bottom: 4px;
  ${e=>e.blocked||e.complete?"opacity: 0.4;":"\n  "}
`,qa=g.default.p`
  color: ${({theme:e})=>e.colorText};
  font-weight: normal;
  font-size: 14px;
  line-height: 22px;
  margin: 0;
`,Ka=g.default.p`
  color: ${({theme:e})=>e.colorText};
  font-weight: 600;
  font-size: 12px;
  line-height: 16px;
  margin: 0;
`;qa.Loud=g.default(qa)`
  font-weight: 600;
`,qa.Quiet=g.default(qa)`
  color: ${({theme:e})=>e.colorTextSecondary};
  ${e=>e.blocked||e.complete?"opacity: 0.4;":"\n  "}
`;var Ga=({stepData:e,style:t={},appearance:n})=>{let{mergeAppearanceWithDefault:r}=Ye(),{primaryCTAClickSideEffects:a}=Ht();n=r(n);let{imageUri:i=null,subtitle:o=null,title:s=null,complete:l=!1,blocked:c=!1}=e;e.primaryButtonTitle||e.secondaryButtonTitle;return m.default.createElement(Ra,{className:ue("carouselCard",n),onClick:c?null:()=>{a(e)},style:t,blocked:c,complete:l},i&&m.default.createElement(Fa,{className:ue("carouselCardImage",n),src:i,alt:s,style:{opacity:l||c?.4:1}}),l&&m.default.createElement(Ba,{className:ue("carouselCompletedPill",n)},m.default.createElement(Ka,{style:{color:"#108E0B"}},"Complete")),s&&m.default.createElement(Wa,{blocked:c,complete:l,className:ue("carouselCardTitle",n),dangerouslySetInnerHTML:yt(s)}),o&&m.default.createElement(qa.Quiet,{blocked:c,complete:l,className:ue("carouselCardSubtitle",n),dangerouslySetInnerHTML:yt(o)}))},Za=()=>m.default.createElement("svg",{width:"24",height:"24",viewBox:"0 0 24 24",fill:"none",xmlns:"http://www.w3.org/2000/svg"},m.default.createElement("path",{d:"M14 6L20 12",stroke:"currentColor",strokeWidth:"2.5",strokeLinecap:"round"}),m.default.createElement("path",{d:"M14 18L20 12",stroke:"currentColor",strokeWidth:"2.5",strokeLinecap:"round"}),m.default.createElement("path",{d:"M4 12H20",stroke:"currentColor",strokeWidth:"2.5",strokeLinecap:"round"})),Ja=({side:e="left",show:t=!1,onClick:n=(()=>{})})=>{let[a,i]=r.useState(!1),[o,s]=r.useState(!1);r.useEffect((()=>{!0===t&&!1===a?i(!0):!1===t&&!0===a&&s(!0)}),[t]);let l="left"==e?{top:0,bottom:0,left:-20,transform:"rotate(180deg)"}:{top:0,bottom:0,right:-20};return a?m.default.createElement(Da,{style:l,reversed:o,onAnimationEnd:o?()=>{i(!1),s(!1)}:null},m.default.createElement(La,{onClick:()=>n(),style:{right:16,top:"calc(50% - 24px)"}},m.default.createElement(Za,null))):null},Xa=({flowId:e,appearance:t,customVariables:n,className:a})=>{let i=r.useRef(null),[o,s]=r.useState(!1),[l,c]=r.useState(!1),[d,u]=r.useState(null),[p,h]=r.useState([]),[f,g]=r.useState(0),{isSmall:y}=(()=>{let e={isSmall:"(max-width: 480px)",isMedium:"(min-width: 481px) AND (max-width: 1023px)",isLarge:"(min-width: 1025px)"},t=Object.fromEntries(Object.entries(e).map((([e])=>[e,!1]))),[n,a]=r.useState(t),i=null,o=()=>{null!==i?clearTimeout(i):s(),i=setTimeout((()=>{s()}),16)},s=()=>{let t=Object.fromEntries(Object.entries(e).map((([e,t])=>{if(!window)return[e,!1];let n=window.matchMedia(t);return n.addEventListener("change",o),[e,n.matches]})));a(t)};return r.useEffect((()=>{s()}),[]),n})(),v=y?1:3,{getFlowMetadata:b,getFlowSteps:x,getNumberOfStepsCompleted:w,updateCustomVariables:_,isLoading:k}=oe();r.useEffect((()=>{_(n)}),[n,k]),r.useEffect((()=>{if(k)return;let t=b(e),n=w(e),r=x(e);u(t),(null!==t.data||null!==t.steps)&&(h(r.sort(((e,t)=>Number(e.complete)-Number(t.complete)))),c(r.length>v),g(n))}),[k]);let S=[];for(let e=0;e<p.length;e+=v)S.push(p.slice(e,e+v));let E=e=>{let t=e.target,n=t.scrollWidth-t.clientWidth,r=Math.ceil(t.scrollLeft);r>0&&!1===o&&s(!0),0===r&&!0===o&&s(!1),r<n&&!1===l&&c(!0),r===n&&!0===l&&c(!1)},A=(e=!0)=>{let t=e?1:-1;null!==i.current&&i.current.scrollBy({left:i.current.clientWidth*t,behavior:"smooth"})},P=null;return k?null:m.default.createElement(Ua,{className:me(ue("carouselContainer",t),a)},m.default.createElement("div",{style:{display:"flex",justifyContent:y?"center":"space-between",marginBottom:20,flexWrap:y?"wrap":"nowrap",gap:y?16:20}},m.default.createElement("div",null,m.default.createElement(Va,{className:ue("carouselTitle",t),dangerouslySetInnerHTML:yt(null==d?void 0:d.title)}),m.default.createElement(qa.Quiet,{className:ue("carouselSubtitle",t),dangerouslySetInnerHTML:yt(null==d?void 0:d.subtitle)})),m.default.createElement(Ha,{className:ue("progressWrapper",t)},m.default.createElement(or,{count:f,total:p.length,appearance:t}))),m.default.createElement("div",{style:{position:"relative"}},m.default.createElement(Ja,{show:o,onClick:()=>A(!1)}),m.default.createElement(Ja,{side:"right",show:l,onClick:A}),m.default.createElement(Ia,{ref:i,onScroll:e=>{null!==P?clearTimeout(P):E(e),P=setTimeout((()=>{E(e)}),16)}},S.map(((e,n)=>m.default.createElement(Ma,{key:n,style:{flex:`0 0 calc(100% - ${p.length>v?36:0}px)`}},e.map(((e,n)=>m.default.createElement(Ga,{key:n,stepData:e,style:{flex:p.length>v?`0 1 calc(100% / ${v} - 16px * 2 / ${v})`:1},appearance:t})))))))),m.default.createElement(Sn,{appearance:t}))},Qa=e=>{var t=e,{flowId:n,title:a,subtitle:i,style:o,initialSelectedStep:s,className:l,type:c="inline",onDismiss:d,visible:u,customVariables:p,onStepCompletion:h,onButtonClick:f,appearance:g,hideOnFlowCompletion:y,setVisible:v,customStepTypes:b,checklistStyle:x="default",autoExpandFirstIncompleteStep:w,autoExpandNextStep:_}=t,k=O(t,["flowId","title","subtitle","style","initialSelectedStep","className","type","onDismiss","visible","customVariables","onStepCompletion","onButtonClick","appearance","hideOnFlowCompletion","setVisible","customStepTypes","checklistStyle","autoExpandFirstIncompleteStep","autoExpandNextStep"]);let{getFlow:S,getFlowSteps:E,markStepCompleted:A,getStepStatus:P,getNumberOfStepsCompleted:C,isLoading:T,targetingLogicShouldHideFlow:I,updateCustomVariables:M,getFlowMetadata:L,isStepBlocked:$,getFlowStatus:R,hasActiveFullPageFlow:U,setHasActiveFullPageFlow:B,markStepStarted:z,getCurrentStepIndex:V}=oe(),{primaryCTAClickSideEffects:H,secondaryCTAClickSideEffects:Y}=Ht(),{getOpenFlowState:W,setOpenFlowState:q}=K(),[G,Z]=r.useState(s||0),[J,X]=r.useState(!1),Q=void 0===u?W(n):u,ee="modal"===c,{mergeAppearanceWithDefault:te}=Ye();En(n,u);let ne=E(n),re=V(n);if(g=te(g),r.useEffect((()=>{M(p)}),[p,T]),r.useEffect((()=>{void 0!==u&&(ee&&!0===u?B(!0):ee&&!1===u&&B(!1))}),[u,v,U]),r.useEffect((()=>{G!==re&&Z(re)}),[re]),T)return null;let ae=S(n);if(!ae||I(ae)||!ne||!0===y&&R(n)===D)return null;let ie=L(n);if(null!=ie&&ie.title&&(a=ie.title),null!=ie&&ie.subtitle&&(i=ie.subtitle),!J&&void 0===s&&C(n)>0){let e=ne.findIndex((e=>!1===e.complete));Z(e>-1?e:ne.length-1),X(!0)}function se(){G+1>=ne.length?ee&&q(n,!1):$(n,ne[G+1].id)||Z(G+1)}function le(e,t,n){let r=G+1<ne.length?ne[G+1]:null;f&&!0===f(e,G,t,r)&&ee&&ue(),h&&h(e,n,r),!h&&!f&&(e.primaryButtonUri||e.secondaryButtonUri)&&ee&&ue()}function ce(){return m.default.createElement(Sn,{appearance:g})}let de={steps:ne.map(((e,t)=>N(j({},e),{handleSecondaryButtonClick:()=>{se(),Y(e),!0===e.skippable&&A(n,e.id,{skipped:!0}),le(e,"secondary",t)},handlePrimaryButtonClick:()=>{(!e.completionCriteria&&(e.autoMarkCompleted||void 0===e.autoMarkCompleted)||e.completionCriteria&&!0===e.autoMarkCompleted)&&(A(n,e.id),se()),le(e,"primary",t),H(e),P(n,e.id)===F&&se()}}))),title:a,subtitle:i,primaryColor:g.theme.colorPrimary,appearance:g,customStepTypes:b,type:c,className:l,autoExpandFirstIncompleteStep:w,autoExpandNextStep:_};function ue(){q(n,!1),d&&d(),v&&v(!1)}function pe(){let e=m.default.createElement(Mr,j({flowId:n,style:o,selectedStep:G,setSelectedStep:Z,appearance:g,type:c},de));return ee?m.default.createElement(Fe,{onClose:()=>{ue()},visible:Q,appearance:g,style:{paddingTop:"0px",padding:"12px",paddingLeft:0}},m.default.createElement(ce,null),e):m.default.createElement(m.default.Fragment,null,m.default.createElement(ce,null),e)}switch(x){case"condensed":return m.default.createElement(m.default.Fragment,null,m.default.createElement(ce,null),m.default.createElement(ja,j({visible:Q,onClose:()=>{ue()},selectedStep:G,setSelectedStep:Z,autoExpandNextStep:!0,appearance:g,onButtonClick:f},de)));case"with-guide":return function(){var e;let t,n=k.guideFlowId;return n&&S(n)&&(t=E(n)),m.default.createElement(m.default.Fragment,null,m.default.createElement(ce,null),m.default.createElement(ha,j({visible:Q,stepsTitle:ie.stepsTitle?ie.stepsTitle:"Your quick start guide",onClose:()=>{ue()},selectedStep:G,setSelectedStep:Z,guideData:t,guideTitle:null!=(e=k.guideTitle)?e:"Guide",appearance:g,title:a,subtitle:i,onGuideButtonClick:e=>{le(e,"link",0)},customStepTypes:b},de)))}();case"default":default:return pe();case"carousel":return m.default.createElement(m.default.Fragment,null,m.default.createElement(ce,null),m.default.createElement(Xa,{flowId:n,appearance:g,customVariables:p,className:l}))}},ei=g.default.div`
  border: 1px solid ${e=>e.appearance.theme.colorBorder};
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
  padding: 10px 12px 10px 12px;
  min-width: 160px;
  cursor: pointer;
  background-color: ${e=>e.appearance.theme.colorBackground}};
  ${e=>fe(e)}
  
  &:hover {
    filter: brightness(.99);
  }
`,ti=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${e=>"condensed"===e.type?"0":"6px"};
  flex-grow: 2;
`,ni=g.default.div`
  text-overflow: ellipsis;
  font-weight: 600;
  font-size: 14px;
  line-height: 16px;
  margin-right: ${e=>"condensed"===e.type?"8px":"0"};
  text-align: ${e=>"condensed"===e.type?"left":"right"};
`,ri=g.default.div`
  width: 20px;
  margin-right: 8px;
  display: flex;
  height: 100%;
  align-items: center;
`,ai=({title:e,count:t,total:n,onClick:r,style:a={},className:i,appearance:o,type:s="default"})=>m.default.createElement(m.default.Fragment,null,m.default.createElement(Sn,{appearance:o}),m.default.createElement(ei,{onClick:()=>void 0!==r&&r(),styleOverrides:j(j({},"condensed"==s?{display:"flex",justifyContent:"space-between"}:{}),a),className:me(null!=i?i:"",ue("progressBadgeContainer",o)),appearance:o},"condensed"==s&&n&&0!==n&&m.default.createElement(ri,{className:ue("progressRingContainer",o)},m.default.createElement(st,{size:19,percentage:t/n,fillColor:o.theme.colorPrimary,bgColor:o.theme.colorBackgroundSecondary})),m.default.createElement(ti,{type:s,className:ue("badgeTitleContainer",o)},m.default.createElement(ni,{type:s,appearance:o,className:ue("badgeTitle",o)},e),void 0!==r&&m.default.createElement(Lr,{className:ue("badgeChevron",o),color:o.theme.colorPrimary})),"default"==s&&n&&0!==n&&m.default.createElement(or,{display:"compact",count:t,total:n,bgColor:o.theme.colorBackgroundSecondary,style:{width:"100%"},appearance:o,textStyle:{color:"#818898"}}))),ii=g.default.div`
  display: flex;
  flex-direction: row;
  width: 100%;
  padding: 16px;
  box-sizing: border-box;
  align-items: center;
  background-color: ${e=>e.appearance.theme.colorBackground};
  border: 1px solid ${e=>e.appearance.theme.colorBorder};
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`,oi=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-right: 16px;
  }
`,si=g.default.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  margin-top: 0;
`,li=g.default.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin-left: 16px;
  min-width: 200px;
`;g.default.div`
  display: flex;
  justify-content: center;
  align-items: flex-end;
  margin-left: 16px;
  cursor: pointer;
  :hover {
    opacity: 0.8;
  }
`;var ci=({title:e,subtitle:t,icon:n,appearance:r,count:a,total:i,className:o,style:s,onClick:l})=>m.default.createElement(m.default.Fragment,null,m.default.createElement(ii,{appearance:r,className:me(ue("fullWidthProgressBadgeContainer",r),null!=o?o:""),style:s,onClick:()=>void 0!==l&&l()},n&&m.default.createElement(oi,{className:ue("fullWidthProgressBadgeIcon",r)},n),m.default.createElement(si,null,m.default.createElement(_t,{size:"small",appearance:r,title:e,subtitle:t})),m.default.createElement(li,{className:ue("fullWidthProgressBadgeProgressContainer",r)},m.default.createElement(or,{count:a,total:i,display:"percent",textLocation:"top"})))),di=(e,t,n,r={x:20,y:20},a)=>{let i="fixed"==a?0:window.scrollY,o="fixed"==a?0:window.scrollX;return"left"===t?{x:e.left-n+r.x+o,y:e.top-r.y+i}:"right"===t?{x:e.left+e.width+r.x+o,y:e.top-r.y+i}:{x:0,y:0}};var ui=12,pi=g.default.div`
  width: 100%;
  height: 100%;
  border-radius: 9999px;
  display: inline-flex;
  background-color: ${e=>e.primaryColor};
  animation-duration: 1.5s;
  animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
  animation-delay: 0.15s;
  animation-iteration-count: infinite;
  animation-direction: normal;
  animation-fill-mode: none;
  animation-play-state: running;
  animation-name: ping;
  opacity: 0.15;

  @keyframes ping {
    75%,
    to {
      transform: scale(1.75);
      opacity: 0;
    }
  }
`,hi=g.default.div`
  width: ${ui}px;
  height: ${ui}px;
  border-radius: 100px;
  background-color: ${e=>e.primaryColor};
  z-index: 20;
  opacity: 1;
`,fi=g.default.div`
  pointer-events: all;
`,mi=g.default.div`
  display: flex;
  align-content: center;
  justify-content: center;
  align-items: center;
  z-index: ${e=>e.zIndex?e.zIndex:90};
`,gi=g.default(mi)`
  width: ${24}px;
  height: ${24}px;
`,yi=({steps:e=[],onDismiss:t,onComplete:n=(()=>{}),tooltipPosition:a="auto",showHighlight:i=!0,primaryColor:o="#000000",offset:s={x:0,y:0},visible:l=!0,containerStyle:c={},selectedStep:d=0,customStepTypes:u,appearance:p,dismissible:h=!1,showHighlightOnly:f,showStepCount:g=!0,completedStepsCount:y=0,showFrigadeBranding:v=!1,cssPosition:b="absolute",onViewTooltip:x,className:w})=>{var _,k,S,E,A,P,C,O,T,I,M,D,L,$,R,F,U,B,z;let{logErrorIfDebugMode:V}=function(){let{debug:e}=r.useContext(Yn),[t,n]=r.useState([]);return{logIfDebugMode:function(r){e&&(t.find((e=>e===r))||n([...t,r]))},logErrorIfDebugMode:function(r){e&&(t.find((e=>e===r))||n([...t,r]))}}}(),[H,Y]=r.useState(),[W,q]=r.useState(new Date),K=r.useRef(null),[G,Z]=r.useState(document.querySelector(e[d].selector)),J=function(e,t){let n="DOMRect"in globalThis?new DOMRect:{height:0,width:0,x:0,y:0,bottom:0,top:0,right:0,left:0,toJSON:()=>{}},[a,i]=r.useState(n),o=r.useCallback((()=>{e&&i(e.getBoundingClientRect())}),[e]);return r.useEffect((()=>(o(),window.addEventListener("resize",o),()=>window.removeEventListener("resize",o))),[e,t]),a}(G,W),[X,Q]=r.useState(),[ee,te]=r.useState(!f),ne="static"!=b?null!=(k=null==(_=e[d])?void 0:_.props)&&k.position?e[d].props.position:b:"static",re=null!=(A=null==(E=null==(S=e[d])?void 0:S.props)?void 0:E.zIndex)?A:90,ae=null!=(P=null==H?void 0:H.width)?P:300,ie=null!=(C=null==H?void 0:H.height)?C:100,[oe,se]=r.useState((new Date).getTime());s="static"!=b?null!=(I=null==(T=null==(O=e[d])?void 0:O.props)?void 0:T.offset)?I:s:{x:0,y:0};let le=window.location.pathname.split("/").pop();r.useLayoutEffect((()=>{K.current&&Y({width:K.current.clientWidth,height:K.current.clientHeight})}),[d,W,ne]),r.useEffect((()=>{f||te(!0)}),[d]),r.useEffect((()=>{l&&ee&&x(d)}),[ee]);let ce=()=>{if("static"===ne)return;let t=document.querySelector(e[d].selector);if(!t)return Q(void 0),Z(null),void V(`FrigadeTour: Could not find element with selector "${e[d].selector}" for step ${e[d].id}`);X&&X===JSON.stringify(null==t?void 0:t.getBoundingClientRect())||(Z(t),q(new Date),t&&Q(JSON.stringify(t.getBoundingClientRect())))};if(r.useEffect((()=>{let e=new MutationObserver(ce);return e.observe(document.body,{subtree:!0,childList:!0}),()=>e.disconnect()}),[ce]),r.useEffect((()=>{let e=new MutationObserver(ce);return e.observe(document.body,{subtree:!0,childList:!0,attributes:!0,attributeFilter:["style","class"]}),()=>e.disconnect()}),[ce]),r.useEffect((()=>{let e=setInterval((()=>{ce()}),10);return()=>clearInterval(e)}),[ce]),r.useLayoutEffect((()=>{setTimeout((()=>{ce()}),500),ce()}),[d,le]),r.useEffect((()=>{if(!l)return;let e=e=>{"Escape"===e.key&&t()};return document.addEventListener("keydown",e),()=>{document.removeEventListener("keydown",e)}}),[]),null===G||!l)return null;let de="auto"===a?"right":a,pe=di(J,de,ae,s,ne),he=J.right+ae>(window.innerWidth||document.documentElement.clientWidth);J.bottom,window.innerHeight||document.documentElement.clientHeight,he&&"auto"===a&&(pe=di(J,"left",ae,s,ne),de="left"),null!=(D=null==(M=e[d])?void 0:M.props)&&D.tooltipPosition&&"auto"!==(null==($=null==(L=e[d])?void 0:L.props)?void 0:$.tooltipPosition)&&("left"===(null==(F=null==(R=e[d])?void 0:R.props)?void 0:F.tooltipPosition)||"right"===(null==(B=null==(U=e[d])?void 0:U.props)?void 0:B.tooltipPosition))&&(de=e[d].props.tooltipPosition);let fe=()=>m.default.createElement(m.default.Fragment,null,g&&e.length>1&&m.default.createElement(je,null,m.default.createElement(Ce,{role:"status",className:ue("tooltipStepCounter",p)},d+1," of ",e.length)),(e[d].primaryButtonTitle||e[d].secondaryButtonTitle)&&m.default.createElement(Ne,{showStepCount:g,className:ue("tooltipCTAContainer",p)},e[d].secondaryButtonTitle&&m.default.createElement(Ut,{title:e[d].secondaryButtonTitle,appearance:p,onClick:()=>{e[d].handleSecondaryButtonClick&&(e[d].handleSecondaryButtonClick(),f&&!e[d].secondaryButtonUri&&te(!1))},size:"small",withMargin:!1,secondary:!0}),e[d].primaryButtonTitle&&m.default.createElement(Ut,{title:e[d].primaryButtonTitle,appearance:p,onClick:()=>{if(e[d].handlePrimaryButtonClick&&(e[d].handlePrimaryButtonClick(),te(!1),setTimeout((()=>{ce()}),30)),y===e.length-1)return n()},withMargin:!1,size:"small"}))),me=()=>{var n,r,a;return m.default.createElement(m.default.Fragment,null,h&&m.default.createElement(ke,{"data-testid":"tooltip-dismiss",onClick:()=>{t&&t()},className:ue("tooltipClose",p),hasImage:!!e[d].imageUri||!!e[d].videoUri,"aria-label":"Close Tooltip",role:"button",tabIndex:0},m.default.createElement(xe,null)),e[d].imageUri&&m.default.createElement(Se,{dismissible:h,appearance:p,role:"img","aria-label":e[d].title,src:e[d].imageUri,className:ue("tooltipImageContainer",p)}),e[d].videoUri&&!e[d].imageUri&&m.default.createElement(Ee,{dismissible:h,appearance:p,role:"video","aria-label":e[d].title,className:ue("tooltipVideoContainer",p)},m.default.createElement(on,{appearance:p,videoUri:e[d].videoUri,autoplay:null==(n=e[d].props)?void 0:n.autoplayVideo,loop:null==(r=e[d].props)?void 0:r.loopVideo,hideControls:null==(a=e[d].props)?void 0:a.hideVideoControls})),m.default.createElement(Pe,{className:ue("tooltipContentContainer",p)},m.default.createElement(_t,{appearance:p,title:e[d].title,subtitle:e[d].subtitle,size:"small",ariaPrefix:`Tooltip${e[d].id}`}),m.default.createElement(Ae,{className:ue("tooltipFooter",p)},m.default.createElement(fe,null))))},ge=j(j({},{default:t=>{var n;if(null!=(n=e[d])&&n.StepContent){let t=e[d].StepContent;return m.default.createElement("div",null,t)}return m.default.createElement(me,null)}}),u);if(!0===e[d].complete||0==pe.x&&0==pe.y&&(new Date).getTime()-oe<100)return null;let ye={top:(null==pe?void 0:pe.y)-ui,left:null!=(z="left"==de?J.x+s.x:(null==pe?void 0:pe.x)-ui)?z:0,cursor:f?"pointer":"default",position:ne},ve=()=>{f&&(q(new Date),te(!ee))};return m.default.createElement(fi,{className:w},m.default.createElement(gi,{style:ye,zIndex:re,className:ue("tourHighlightContainer",p)},i&&!1!==e[d].showHighlight&&"static"!==b&&m.default.createElement(m.default.Fragment,null,m.default.createElement(hi,{style:{position:ne},onClick:ve,primaryColor:p.theme.colorPrimary,className:ue("tourHighlightInnerCircle",p)}),m.default.createElement(pi,{style:{position:"absolute"},onClick:ve,primaryColor:p.theme.colorPrimary,className:ue("tourHighlightOuterCircle",p)}))),m.default.createElement(mi,{style:N(j({},ye),{left:(()=>{let e=ye.left+("left"==de?-ae:24);return Math.min(Math.max(e,20),window.innerWidth-ae-20)})(),top:ye.top+ie>window.innerHeight-20?ye.top+-ie:ye.top}),zIndex:re+1,className:ue("tooltipContainerWrapper",p)},ee&&m.default.createElement(m.default.Fragment,null,m.default.createElement(_e,{ref:K,role:"dialog","aria-labelledby":`frigadeTooltip${e[d].id}Title`,"aria-describedby":`frigadeTooltip${e[d].id}Subtitle`,layoutId:"tooltip-container",tabIndex:0,"aria-label":"Tooltip",style:j({position:"relative",width:"max-content",right:0,top:"static"!==b?12:0},c),appearance:p,className:ue("tooltipContainer",p),maxWidth:300,zIndex:re+10},m.default.createElement((()=>{var t;return e?null!=(t=e[d])&&t.type&&ge[e[d].type]?ge[e[d].type]({stepData:e[d],primaryColor:o}):ge.default(e[d]):m.default.createElement(m.default.Fragment,null)}),null)),v&&m.default.createElement(Te,{className:ue("poweredByFrigadeTooltipRibbon",p),appearance:p,zIndex:re+10},m.default.createElement(Me,{appearance:p})))))},vi=g.default.button`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class

    display: flex;
    flex-direction: row;
    justify-content: space-between;
    padding: 6px 10px;
    gap: 8px;

    background: #fafafa;
    border: 1px solid #d9d9d9;
    border-radius: 21px;
    font-size: 12px;
    :hover {
      opacity: 0.8;
    }
  }
`,bi=g.default.span`
  ${e=>pe(e)} {
    font-size: 12px;
    display: inline-block;
  }
`,xi=g.default.span`
  ${e=>pe(e)} {
    font-size: 12px;
    display: inline-block;
  }
`,wi=g.default.div`
  position: fixed;
  right: 0;
  bottom: 0;
  margin-right: 24px;
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  z-index: 50;
`,_i=g.default.button`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background-color: #ffffff;
    border: 1px solid #f5f5f5;
  }
  width: 50px;
  height: 50px;
  display: flex;
  align-content: center;
  align-items: center;
  justify-content: center;
  box-shadow: 0px 9px 28px 8px rgba(0, 0, 0, 0.05), 0px 6px 16px rgba(0, 0, 0, 0.08),
    0px 3px 6px -4px rgba(0, 0, 0, 0.12);
  border-radius: 45px;
  cursor: pointer;
`,ki=g.default.div`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    background: #ffffff;
  }

  display: flex;
  flex-direction: column;
  min-width: 200px;
  padding: 4px;
  box-shadow: 0px 9px 28px 8px rgba(0, 0, 0, 0.05), 0px 6px 16px rgba(0, 0, 0, 0.08),
    0px 3px 6px -4px rgba(0, 0, 0, 0.12);
  border-radius: 8px;
  margin-bottom: 22px;
  position: ${e=>"inline"==e.type?"absolute":"relative"};
  top: ${e=>"inline"==e.type?"68px":0};
  margin-left: ${e=>"inline"==e.type?"-127px":0};
`,Si=g.default.button`
  ${e=>pe(e)} {
    // Anything inside this block will be ignored if the user provides a custom class
    color: #434343;
    :hover {
      background-color: #f5f5f5;
    }
  }

  display: flex;
  border-radius: 8px;
  background-color: transparent;
  border: none;
  cursor: pointer;
  font-size: 14px;

  padding: 8px 12px;
`,Ei=({style:e,className:t})=>m.default.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",width:"18",height:"18",fill:"none",viewBox:"0 0 18 18",style:e,className:t},m.default.createElement("path",{fill:"currentColor",d:"M13.43 4.938a4.494 4.494 0 00-1.043-1.435A4.955 4.955 0 009 2.197c-1.276 0-2.48.464-3.387 1.305A4.502 4.502 0 004.57 4.938a4.242 4.242 0 00-.386 1.773v.475c0 .109.087.197.196.197h.95a.197.197 0 00.197-.197V6.71c0-1.749 1.557-3.17 3.473-3.17s3.473 1.421 3.473 3.17c0 .718-.254 1.393-.738 1.955a3.537 3.537 0 01-1.9 1.125 1.928 1.928 0 00-1.085.682c-.271.343-.42.768-.42 1.206v.552c0 .109.088.197.197.197h.95a.197.197 0 00.196-.197v-.552c0-.276.192-.519.457-.578a4.904 4.904 0 002.625-1.56c.335-.392.597-.828.778-1.3a4.256 4.256 0 00-.103-3.303zM9 13.834a.985.985 0 10.001 1.97.985.985 0 00-.001-1.97z"})),Ai=({style:e,className:t})=>m.default.createElement("svg",{style:e,className:t,xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:"1.5",stroke:"currentColor"},m.default.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z"})),Pi=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 400px;
    padding: 28px 18px;
    box-sizing: border-box;
    align-items: unset;
    background-color: ${e=>e.appearance.theme.colorBackground};
    border-width: 1px;
    border-color: ${e=>e.appearance.theme.colorBorder};
    border-radius: 12px;
    position: relative;
  }
`,ji=g.default.div`
  display: flex;
  flex-direction: column;
  flex: 1;
`,Ni=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: flex-start;
  align-items: center;
  margin-top: 16px;
  gap: 12px;
`,Ci=g.default.div`
  ${e=>pe(e)} {
    position: absolute;
    top: 16px;
    right: 16px;
    cursor: pointer;

    :hover {
      opacity: 0.8;
    }
  }
`,Oi=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 500px;
    box-sizing: border-box;
    align-items: unset;
    background-color: ${e=>e.appearance.theme.colorBackground};
    position: relative;
  }
`,Ti=g.default.div`
  display: flex;
  flex-direction: column;
  flex: 1;
`,Ii=g.default.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-bottom: 12px;
  margin-top: 4px;
`,Mi=g.default.div`
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: stretch;
  gap: 16px;
  margin-top: 8px;
`,Di=g.default.div`
  margin-top: 16px;
  margin-bottom: 16px;
`,Li=g.default.div`
  ${e=>pe(e)} {
    position: absolute;
    top: -16px;
    right: -16px;
    cursor: pointer;

    :hover {
      opacity: 0.8;
    }
  }
`,$i=g.default.h1`
  ${e=>pe(e)} {
    font-style: normal;
    justify-content: center;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    line-height: 24px; /* 125% */
    letter-spacing: 0.36px;
    display: flex;
    align-items: center;
    color: ${e=>e.appearance.theme.colorText};
    margin-bottom: 8px;
  }
`,Ri=g.default.h2`
  ${e=>pe(e)} {
    font-style: normal;
    justify-content: center;
    text-align: center;
    font-weight: 400;
    color: ${e=>e.appearance.theme.colorTextSecondary};
    font-size: 14px;
    line-height: 22px; /* 150% */
    letter-spacing: 0.28px;
    margin-bottom: 8px;
  }
`,Fi=g.default.img`
  width: 100%;
  height: 100%;
  min-height: 200px;
  border-radius: ${e=>e.appearance.theme.borderRadius}px;
`;function Ui({stepData:e,appearance:t,classPrefix:n=""}){var r,a,i;return e.videoUri?m.default.createElement(on,{appearance:t,videoUri:e.videoUri,autoplay:null==(r=e.props)?void 0:r.autoplayVideo,loop:null==(a=e.props)?void 0:a.loopVideo,hideControls:null==(i=e.props)?void 0:i.hideVideoControls}):e.imageUri?m.default.createElement(Fi,{className:ue(`${n}image`,t),appearance:t,src:e.imageUri}):null}var Bi=g.default.div`
  // use the :not annotation
  ${e=>pe(e)} {
    display: flex;
    flex-direction: ${e=>"square"===e.type?"column":"row"};
    width: 100%;
    padding: 16px;
    box-sizing: border-box;
    align-items: ${e=>"square"===e.type?"unset":"center"};
    background-color: ${e=>e.appearance.theme.colorBackground};
    border-radius: 12px;
  }
`,zi=g.default.div`
  ${e=>pe(e)} {
    display: flex;
    width: 46px;
    height: 46px;
  }
`,Vi=g.default.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  margin-left: ${e=>"square"===e.type?"0px":"16px"};
  margin-top: ${e=>"square"===e.type?"12px":"0"};
`,Hi=g.default.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin-left: ${e=>"square"===e.type?"0px":"16px"};
`,Yi=g.default.div`
  display: flex;
  justify-content: ${e=>"square"===e.type?"flex-end":"center"};
  align-items: flex-end;
  cursor: pointer;
  :hover {
    opacity: 0.8;
  }
`,Wi=g.default.div`
  display: flex;
  justify-content: ${e=>"square"===e.type?"flex-end":"center"};
  align-items: flex-end;
  margin-left: ${e=>"square"===e.type?"0px":"16px"};
`,qi=({style:e,className:t})=>m.default.createElement("svg",{xmlns:"http://www.w3.org/2000/svg",width:"46",height:"46",fill:"none",viewBox:"0 0 46 46",style:e,className:t},m.default.createElement("circle",{cx:"23",cy:"23",r:"23",fill:"#E6F1FF"}),m.default.createElement("path",{stroke:"#0171F8",strokeLinecap:"round",strokeLinejoin:"round",strokeWidth:"1.5",d:"M32 18.5l-2.25-1.313M32 18.5v2.25m0-2.25l-2.25 1.313M14 18.5l2.25-1.313M14 18.5l2.25 1.313M14 18.5v2.25m9 3l2.25-1.313M23 23.75l-2.25-1.313M23 23.75V26m0 6.75l2.25-1.313M23 32.75V30.5m0 2.25l-2.25-1.313m0-16.875L23 13.25l2.25 1.313M32 25.25v2.25l-2.25 1.313m-13.5 0L14 27.5v-2.25"})),Ki=g.default.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 600px;
  padding: 28px 18px;
  box-sizing: border-box;
  align-items: unset;
  background-color: ${e=>e.appearance.theme.colorBackground};
  position: ${e=>"modal"==e.type?"fixed":"relative"};

  min-width: 550px;

  ${e=>"modal"==e.type?`\n  left: 50%;\n  transform: translate(-50%);\n  bottom: 24px;\n  z-index: 1000;\n  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);\n  border-width: 1px;\n  border-color: ${e=>e.appearance.theme.colorBorder};\n  border-radius: ${e=>e.appearance.theme.borderRadius}px;`:""}
`,Gi=g.default.button`
  border: 1px solid ${e=>e.appearance.theme.colorBorder};
  border-radius: 8px;
  // If selected make border color primary and text color color priamry
  border-color: ${e=>e.selected?e.appearance.theme.colorPrimary:e.appearance.theme.colorBorder};
  color: ${e=>e.selected?e.appearance.theme.colorPrimary:e.appearance.theme.colorText};
  :hover {
    border-color: ${e=>e.appearance.theme.colorPrimary};
  }
  :focus {
    border-color: ${e=>e.appearance.theme.colorPrimary};
    color: ${e=>e.appearance.theme.colorPrimary};
  }
  font-size: 16px;
  font-weight: 600;
  line-height: 24px;
  width: 44px;
  height: 44px;
  display: flex;
  justify-content: center;
  align-items: center;
`,Zi=g.default.div`
  display: flex;
  justify-content: space-between;
  margin-top: 16px;
  gap: 8px;
`,Ji=g.default.div`
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
`,Xi=g.default.div`
  font-size: 12px;
  line-height: 18px;
  color: ${e=>e.appearance.theme.colorTextDisabled};
  font-style: normal;
  font-weight: 400;
  letter-spacing: 0.24px;
`,Qi=g.default.div`
  display: flex;
  flex-direction: column;
  flex: 1;
`;g.default.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  margin-top: 16px;
`;var eo=g.default.textarea`
  ${e=>pe(e)} {
    color: ${e=>e.appearance.theme.colorText};
    margin-top: 16px;
    border: 1px solid ${e=>e.appearance.theme.colorBorder};
    border-radius: ${e=>e.appearance.theme.borderRadius}px;
    padding: 12px 16px;
    font-size: 16px;
    line-height: 24px;
    width: 100%;
    height: 100px;
    resize: none;
  }
`,to=g.default.div`
  ${e=>pe(e)} {
    position: absolute;
    top: 16px;
    right: 16px;
    cursor: pointer;

    :hover {
      opacity: 0.8;
    }
  }