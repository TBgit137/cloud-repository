const BASE = 'http://127.0.0.1:8000'

/**
 * 上传数据集，通过 SSE 流接收处理进度和最终结果。
 *
 * @param {File} file
 * @param {Object} params - { airport, nRunways, safetyInterval, startDate, endDate }
 * @param {Function} onStatus - (step: string, message: string) => void  每个阶段回调
 * @returns {Promise<{summary, schedule, removed_flights}>}  resolve 于 done 事件
 */
export function uploadDataset(file, params, onStatus) {
  const form = new FormData()
  form.append('file', file)
  form.append('airport', params.airport || 'SBGR')
  form.append('n_runways', params.nRunways || 5)
  form.append('safety_interval', params.safetyInterval || 3)
  form.append('start_date', params.startDate || '')
  form.append('end_date', params.endDate || '')

  return new Promise(async (resolve, reject) => {
    let res
    try {
      res = await fetch(`${BASE}/api/upload`, { method: 'POST', body: form })
    } catch (e) {
      return reject(new Error('无法连接到服务器'))
    }

    if (!res.ok) {
      // 非流式错误（如 422 验证失败在流开始前）
      try {
        const data = await res.json()
        return reject(new Error(data.message || '请求失败'))
      } catch {
        return reject(new Error('请求失败'))
      }
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    const pump = async () => {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // SSE 消息以 \n\n 分隔
        const parts = buffer.split('\n\n')
        buffer = parts.pop() // 最后一段可能不完整，留到下次

        for (const part of parts) {
          if (!part.trim()) continue

          let eventName = 'message'
          let dataStr = ''

          for (const line of part.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim()
            else if (line.startsWith('data: ')) dataStr = line.slice(6).trim()
          }

          if (!dataStr) continue
          let payload
          try { payload = JSON.parse(dataStr) } catch { continue }

          if (eventName === 'status') {
            onStatus?.(payload.step, payload.message)
          } else if (eventName === 'error') {
            return reject(new Error(payload.message))
          } else if (eventName === 'done') {
            return resolve(payload)
          }
        }
      }
    }

    pump().catch(reject)
  })
}
