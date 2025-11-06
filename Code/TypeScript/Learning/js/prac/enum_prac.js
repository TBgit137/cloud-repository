"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var AuditStatus;
(function (AuditStatus) {
    AuditStatus[AuditStatus["MANAGER_AUDIT_FAIL"] = -1] = "MANAGER_AUDIT_FAIL";
    AuditStatus[AuditStatus["NO_AUDIT"] = 0] = "NO_AUDIT";
    AuditStatus[AuditStatus["MANAGER_AUDIT_SUCCESS"] = 1] = "MANAGER_AUDIT_SUCCESS";
    AuditStatus[AuditStatus["FINAL_AUDIT_SUCCESS"] = 2] = "FINAL_AUDIT_SUCCESS";
})(AuditStatus || (AuditStatus = {}));
class Audit {
    /**
     * getAuditState
     */
    getAuditState(status) {
        if (status === AuditStatus.MANAGER_AUDIT_FAIL) {
            console.log("经理审核失败");
        }
        else if (status === AuditStatus.NO_AUDIT) {
            console.log("没有审核");
        }
        else if (status === AuditStatus.MANAGER_AUDIT_SUCCESS) {
            console.log("经理审核通过");
        }
        else if (status === AuditStatus.FINAL_AUDIT_SUCCESS) {
            console.log("财务审核通过");
        }
    }
}
