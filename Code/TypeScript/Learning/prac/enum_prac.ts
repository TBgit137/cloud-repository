enum AuditStatus{
    MANAGER_AUDIT_FAIL = -1,
    NO_AUDIT,
    MANAGER_AUDIT_SUCCESS,
    FINAL_AUDIT_SUCCESS
}

class Audit{
    /**
     * getAuditState
     */
    public getAuditState(status: AuditStatus) {
        if(status === AuditStatus.MANAGER_AUDIT_FAIL){
            console.log("经理审核失败");
        } else if(status === AuditStatus.NO_AUDIT){
            console.log("没有审核");
        } else if(status === AuditStatus.MANAGER_AUDIT_SUCCESS){
            console.log("经理审核通过");
        } else if(status === AuditStatus.FINAL_AUDIT_SUCCESS){
            console.log("财务审核通过");
        }
    }
}