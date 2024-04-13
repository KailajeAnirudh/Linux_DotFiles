## Observations 12-01
### Angular momentum constraint calc

### System specific changes

### Solver not working pal!!!
 - *Observation*: controls and costs blowing up at jump-aerial phase switch. 
 - Diagnosis: Logging showed feet start to slip at the take off moment. Phase switch detection was delayed, as a result phase 1 controller was working when phase 2 was active.